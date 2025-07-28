# simulation_optimized.jl

using CUDA
using StaticArrays
using LinearAlgebra
using Flux
using Logging

# 1) Daten-Generierung pro Batch (Float32, GPU-only)
function only_times_and_dist_gpu(
    batch_size::Int;
    ear_positions::Vector{SVector{3,Float32}} = [
      SVector{3,Float32}(1,0,0),
      SVector{3,Float32}(0,1,0),
      SVector{3,Float32}(0,0,1),
      SVector{3,Float32}(0,0,0)
    ],
    speed_sound::Float32 = 343f0,
    mic_rate::Int       = 44_000,
    dt::Float32         = 1f0/mic_rate,
    num_ears = length(ear_positions))

    # Positionen: 3 × batch_size auf GPU
    positions = CUDA.rand(Float32, 3, batch_size) .* 200f0 .- 100f0

    # Ear-Positions: 3 × num_ears auf GPU
    ep_cpu = hcat(ear_positions...)
    ep     = CUDA.CuArray{Float32}(ep_cpu)

    # Distanzen: num_ears × batch_size
    distances = CUDA.zeros(Float32, num_ears, batch_size)
    for n in 1:num_ears
        sqsum = CUDA.sum((positions .- ep[:, n]).^2; dims=1)        # 1×batch
        vec   = CUDA.dropdims(sqsum; dims=1)                        # batch-Vektor
        distances[n, :] .= sqrt.(vec)
    end

    # Volumen-Kanal: 1 / dist^2
    inv2 = distances .^ -2                                           # num_ears×batch

    # Zeit-Kanal: ceil((t - t_min) / dt)
    raw_times = distances ./ speed_sound                             # num_ears×batch
    mins      = CUDA.minimum(raw_times; dims=1)                      # 1×batch
    times     = CUDA.ceil.((raw_times .- mins) ./ dt) .*dt                # num_ears×batch

    # Daten zusammenfügen: (2*num_ears) × batch_size
    data = vcat(inv2, times)

    return data, positions
end

# 2) Trainingsroutine mit Flux.Data.DataLoader
function do_ki_only_times_actual_batches(
    batch_size::Int,
    batches_per_epoch::Int,
    num_ears::Int;
    epochs::Int      = 10,
    new_data::Int    = 5,
    print_every::Int = max(1, batches_per_epoch ÷ 10),
    eval_b_size::Int = 100
)
    device = Flux.gpu_device()
    # a) Modell in Float32 auf GPU
    model = Flux.Chain(
        Flux.Dense(2*num_ears, 500*num_ears, Flux.relu),
        Flux.Dense(500*num_ears, 500*num_ears, Flux.relu),
        Flux.Dense(500*num_ears, 500*num_ears, Flux.relu),
        Flux.Dense(500*num_ears, 500*num_ears, Flux.tanhshrink),
        Flux.Dense(500*num_ears, 200*num_ears, Flux.tanhshrink),
        Flux.Dense(200*num_ears, 100*num_ears, Flux.tanhshrink),
        Flux.Dense(100*num_ears, 3)
    ) |> device |> Flux.f32

    # b) Optimizer
    opt = Flux.setup(Flux.NADAM(), model)

    # c) Initiale Daten und Loader
    total       = batch_size * batches_per_epoch
    data_learn, pos_learn = only_times_and_dist_gpu(total) |> device
    data_test,  pos_test  = only_times_and_dist_gpu(eval_b_size) |> device

    loader = Flux.DataLoader((data_learn, pos_learn);
        batchsize = batch_size,
        shuffle   = false)

    # Aufzeichnungen
    train_losses = Float32[]
    test_losses  = Float32[]

    for epoch in 1:epochs
        # neue Daten alle `new_data` Epochen
        if epoch % new_data == 0
            data_learn, pos_learn = only_times_and_dist_gpu(total; num_ears=num_ears) |> device
            loader = Flux.DataLoader((data_learn, pos_learn);
                batchsize = batch_size,
                shuffle   = false)
        end

        # Training über Batches
        for (i, (xb, yb)) in enumerate(loader)
            grads = Flux.gradient(model) do m
                ŷ = m(xb)
                Flux.Losses.mse(ŷ .*100, yb)
            end
            Flux.update!(opt, model, grads[1])

            if i % print_every == 0
                @info "Epoch $epoch — Batch $i/$(batches_per_epoch)"
            end
        end

        # Verluste am Ende der Epoche
        train_loss = Flux.Losses.mae(model(data_learn[:,1:eval_b_size,:]) .*100, pos_learn[:,1:eval_b_size])
        test_loss  = Flux.Losses.mae(model(data_test) .*100,  pos_test)
        push!(train_losses, train_loss)
        push!(test_losses,  test_loss)
        @info "Epoch $epoch done: train_mae=$(round(train_loss,digits=4)) test_mae=$(round(test_loss,digits=4))"
    end

    return train_losses, test_losses
end

# 3) Plot-Funktion
function plot_accuracy(train_losses, test_losses, filename::String)
    fig = Makie.Figure()
    ax  = Makie.Axis(fig[1,1],
        title  = "MAE over Epochs",
        xlabel = LaTeXStrings.L"\mathrm{Epoch}",
        ylabel = LaTeXStrings.L"\mathrm{MAE}"
    )
    Makie.lines!(ax, 1:length(train_losses), train_losses, label="Train")
    Makie.lines!(ax, 1:length(test_losses),  test_losses,  label="Test")
    Makie.axislegend(ax)
    CairoMakie.save(filename * ".png", fig)
end

# 4) Main
function main()
    batch_size        = 10_000
    batches_per_epoch = 100
    num_ears          = 4
    epochs            = 250

    train_losses, test_losses = do_ki_only_times_actual_batches(
        batch_size,
        batches_per_epoch,
        num_ears;
        epochs      = epochs,
        new_data    = 100,
        print_every = 20,
        eval_b_size = 100
    )

    plot_accuracy(train_losses, test_losses,
        joinpath(@__DIR__, "accuracy_gpu"))
    println("Training complete!")
end

# Skript ausführen
main()
