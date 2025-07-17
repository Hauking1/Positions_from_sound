import Makie
import LaTeXStrings
import CairoMakie
import Flux
import LinearAlgebra
import StaticArrays


function create_batch_signals_full_data(batch_size_create_data::Int, listening_length::Int; mic_rate::Int=44000, dt::Float64=1/mic_rate)
    rand_float_0_1 = rand(Float64,3*batch_size_create_data)
    results = zeros(batch_size_create_data,listening_length)
    for index in range(0,batch_size_create_data-1)
        results[index+1,:] = rand_float_0_1[3*index+1]*sin.(rand_float_0_1[3*index+2]*20_000* dt *(0:listening_length-1) .+ rand_float_0_1[3*index+3])
    end
    return results
end

function save_signals_full_data(saving_data_path::String,signals)
    open(saving_data_path,"w") do file
        write.(file,join.(eachrow(signals),",").*"\n")
    end
end

function wrapper_create_save_data_full_data(batch_size_create_data::Int,listening_length::Int,saving_data_path::String)
    save_signals_full_data(saving_data_path,create_batch_signals_full_data(batch_size_create_data,listening_length))
end

function prepare_data_full_learn(data;ear_positions=[[1,0,0],[0,1,0],[0,0,1]],speed_sound = 343., mic_rate::Int=44000 , dt::Float64=1/mic_rate)
    batch_size, listening_length = size(data)
    positions_sound = rand(3*batch_size)*200 .- 100

    @inbounds distances = [LinearAlgebra.norm.([positions_sound[3*index+1:3*index+3] .- ear_positions[1],positions_sound[3*index+1:3*index+3] .- ear_positions[2],positions_sound[3*index+1:3*index+3] .- ear_positions[3]]) for index in range(0,batch_size-1)]
    times = [ceil.(Int,(dist/speed_sound .- minimum(dist/speed_sound))/dt) for dist in distances]
    
    results = [[[0. for _ in 1:listening_length] for _ in 1:3] for _ in 1:batch_size]
    @inbounds for index in range(1,batch_size)
        mic_data = [let shifted = (1/distances[index][n_ear]^2)*circshift(data[index,:],times[index][n_ear])
        shifted[1:times[index][n_ear]] .= 0
        shifted
        end for n_ear in 1:3]
        results[index] = mic_data 
    end
    return results,positions_sound
end

function faster_prepare_data_full_learn(data;ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0),(0,1,0),(0,0,1)),speed_sound = 343., mic_rate::Int=44000 , dt::Float64=1/mic_rate)
    batch_size, listening_length = size(data)
    positions_sound = rand(3*batch_size)*200 .- 100

    @inbounds distances = [@StaticArrays.SVector [LinearAlgebra.norm(positions_sound[3*index+1:3*index+3] .- ear_positions[n_ear]) for n_ear in 1:3] for index in range(0,batch_size-1)]
    times = [ceil.(Int,(dist/speed_sound .- minimum(dist/speed_sound))/dt) for dist in distances]
    
    results = [[Vector{Float64}(undef, listening_length) for _ in 1:3] for _ in 1:batch_size]
    @inbounds for index in range(1,batch_size)
        row = @view data[index, :]
        @inbounds for n_ear in 1:3
            circshift!(results[index][n_ear],row,times[index][n_ear])
            results[index][n_ear] .*= (1/distances[index][n_ear]^2)
            @inbounds results[index][n_ear][1:times[index][n_ear]] .= 0
        end
    end
    return results,positions_sound
end

function less_d_faster_prepare_data_full_learn(data;ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0),(0,1,0),(0,0,1)),speed_sound = 343., mic_rate::Int=44000 , dt::Float64=1/mic_rate)
    batch_size, listening_length = size(data)
    positions_sound = StaticArrays.rand(3*batch_size) .*200 .- 100

    @inbounds distances = [@StaticArrays.SVector [LinearAlgebra.norm(positions_sound[3*index+1:3*index+3] .- ear_positions[n_ear]) for n_ear in 1:3] for index in range(0,batch_size-1)]
    times = [ceil.(Int,(dist/speed_sound .- minimum(dist/speed_sound))/dt) for dist in distances]
    
    results = [Vector{Float64}(undef, 3*listening_length) for _ in 1:batch_size]

    @inbounds for index in range(1,batch_size)
        row = @view data[index, :]
        @inbounds for n_ear in 1:3
            l_index = (n_ear-1)*listening_length+1
            u_index = n_ear*listening_length
            res_write_to = @view results[index][l_index:u_index]
            circshift!(res_write_to,row,times[index][n_ear])
            @inbounds results[index][l_index:u_index] .*= (1/distances[index][n_ear]^2)
            @inbounds results[index][l_index:l_index+times[index][n_ear]] .= 0    #
        end
    end
    return results,permutedims(reshape(positions_sound,(batch_size,3)),[2, 1])
end




function create_save_batch_signal_encoded(batch_size_create_data::Int,saving_data_path::String)
    open(saving_data_path,"w") do file
        write.(file,join.(eachrow(rand(Float64,batch_size_create_data,6)),",").*"\n")
    end
    return nothing
end

struct Data_reader_organiser
    max_num_reads::Int
    current_read::Int
end

function read_encoded_data_to_measurement(saving_data_path::String,learn_batch_size::Int,listening_length::Int, sound_position ;ear_positions=[[1,0,0],[0,1,0],[0,0,1]],speed_sound = 343.)
    data_to_learn = zeros(learn_batch_size,3*listening_length)
    open(saving_data_path,"r") do file
        for index in range(1,learn_batch_size)

        end
    end
end

function wrapper_read_write()
    
end

saving_plot_path = (@__DIR__)*"/plots/"
saving_data_path = (@__DIR__)*"/data/"
saving_data_to = saving_data_path*"train_batch.txt"

println("HI :)")

batch_size_create_data = 1_00
listening_length = 44_00

#@time wrapper_create_save_data_full_data(batch_size_create_data,listening_length,saving_data_to)

#@time data_learn,positions = faster_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))

@time data_learn,positions = less_d_faster_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))

#@time data_learn,positions = prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))



model = Flux.Chain(Flux.Dense(3*listening_length=>500,Flux.relu),
    Flux.Dense(500=>100,Flux.relu),
    Flux.Dense(100=>3,Flux.relu))
Flux.f64(model)


opt_state = Flux.setup(Flux.Adam(), model)

for epoch in 1:10
    for index in 1:batch_size_create_data
    # Calculate the gradient of the objective
    # with respect to the parameters within the model:
    grads = Flux.gradient(model) do m
        result = m(data_learn[index])
        Flux.Losses.mse(result, positions[index])
    end

    # Update the parameters so as to reduce the objective,
    # according the chosen optimisation rule:
    Flux.update!(opt_state, model, grads[1])
    print("\r")
    print("Epoch: $epoch Advanced: $(round(index/batch_size_create_data*100,digits=1))%")
    end
    global opt_state = Flux.setup(Flux.Adam(1/(2*epoch)), model)
end

#@time data_test,positions_test = less_d_faster_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))

g_mse = 0.
for index in 1:batch_size_create_data
    global g_mse +=Flux.Losses.mse(model(data_learn[index]),positions[index])/batch_size_create_data
end
println()
println("the average mse is: $g_mse")


function plot_concatenated_dat(data_learn,number_data::Int)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1],title = "many signals",
    xlabel = LaTeXStrings.LaTeXString("time/dt"),
    ylabel = LaTeXStrings.LaTeXString("signal(t)"))
    for index_ear in range(1,number_data)
        Makie.lines!(ax, data_learn[index_ear],label="signal concatinated: $index_ear")
    end
    Makie.axislegend()
    CairoMakie.display(fig)
    CairoMakie.save(saving_plot_path*"Test_less_d.png",fig)
end

function plot_single_ear_data(data_learn)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1],title = "many signals",
    xlabel = LaTeXStrings.LaTeXString("time/dt"),
    ylabel = LaTeXStrings.LaTeXString("signal(t)"))
    for index_ear in range(1,3)
        Makie.lines!(ax, data_learn[1][index_ear],label="signal ear: $index_ear")
    end
    Makie.axislegend()
    CairoMakie.display(fig)
    CairoMakie.save(saving_plot_path*"Test_single_ear.png",fig)
end

#plot_concatenated_dat(data_learn,1)

#plot_single_ear_data(data_learn)

print("done")

