import Makie
import LaTeXStrings
import CairoMakie
import Flux
import LinearAlgebra
import StaticArrays


function only_times_and_dist(batch_size; ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0,0),(0,1,0,0),(0,0,1,0)), speed_sound = 343., mic_rate::Int=44000, dt::Float64=1/mic_rate, num_ears=length(ear_positions))
    positions_sound = StaticArrays.rand(3*batch_size) .*200 .- 100
    
    results = [Vector{Float64}(undef, 2*num_ears) for _ in 1:batch_size]
    #volumes = rand(batch_size)

    @inbounds for index in 1:batch_size
        distances = [LinearAlgebra.norm(positions_sound[3*(index-1)+1:3*index] .- ear_positions[n_ear]) for n_ear in 1:num_ears]
        results[index][1:num_ears] .= [1/(dist^2) for dist in distances] #volumes[index]
        results[index][num_ears+1:end] .= ceil.(Int,(distances/speed_sound .- minimum(distances/speed_sound))/dt)
    end
    return results,positions_sound
end


function do_ki_only_times_actual_batches(batch_size_create_data, batches_per_epoch,num_ears ;epochs=2,new_data = 5, print_every = batches_per_epoch//1000, evaluation_batch_size = batch_size_create_data)
    @time data_learn,positions = only_times_and_dist(batch_size_create_data*batches_per_epoch)

    model = Flux.Chain(Flux.Dense(2*num_ears=>50*num_ears,Flux.tanhshrink),Flux.Dense(50*num_ears=>3))
    
    #model = Flux.Chain(Flux.Dense(2*num_ears=>3),Flux.tanhshrink) #best model yet
    #model = Flux.Chain(Flux.Dense(2*num_ears=>5*num_ears,Flux.tanhshrink),Flux.Dense(5*num_ears=>3))  #with prunin ok
    #model = Flux.Chain(Flux.Dense(2*num_ears=>25*num_ears,Flux.tanhshrink),Flux.Dense(25*num_ears=>25*num_ears,Flux.tanhshrink),Flux.Dense(25*num_ears=>3))



    model = Flux.f64(model)

    opt_state = Flux.setup(Flux.Adam(), model)
    #opt_state = Flux.setup(Flux.Descent(), model)
    #opt_state = Flux.setup(Flux.Momentum(), model)
    #opt_state = Flux.setup(Flux.NADAM(), model)


    train_accuracies = zeros(epochs)
    test_accuracies = zeros(epochs)

    data_test,positions_test = only_times_and_dist(evaluation_batch_size)


    for epoch in 1:epochs

        if epoch%new_data==0
            data_learn,positions = only_times_and_dist(batch_size_create_data*batches_per_epoch)
        end
        
        #model = Flux.trainmode!(model)

        for index in 1:batches_per_epoch
            batch_start = (index - 1) * batch_size_create_data + 1
            batch_end = index * batch_size_create_data
            raw_batch = data_learn[batch_start:batch_end]
            data_batch = reduce(hcat, raw_batch)
            positions_batch = positions[3*(batch_start - 1) + 1 : 3*batch_end]
            positions_batch = reshape(positions_batch, 3, :)
            grads = Flux.gradient(model) do m
                result = m(data_batch)
                Flux.Losses.mse(result, positions_batch)    # L1 pruning now slower
            end
            Flux.update!(opt_state, model, grads[1])

            

            if index%print_every==0
                print("\r")
                print("Epoch: $epoch Advanced: $(round(index/(batches_per_epoch)*100,digits=1))%")
            end
        end

        
        g_mse_train = 0.
        g_mse_test = 0.
        for index in 1:evaluation_batch_size
            g_mse_train += Flux.Losses.mae(model(data_learn[index]),positions[3*(index-1)+1:3*index])/evaluation_batch_size#sqrt(sum((model(data_learn[index]) .- positions[3*(index-1)+1:3*index]).^2))/evaluation_batch_size
            g_mse_test += Flux.Losses.mae(model(data_test[index]),positions_test[3*(index-1)+1:3*index])/evaluation_batch_size#sqrt(sum((model(data_test[index]) .- positions_test[3*(index-1)+1:3*index]).^2))/evaluation_batch_size
        end
        train_accuracies[epoch] = g_mse_train
        test_accuracies[epoch] = g_mse_test
        
        print("\r")
        println("Epoch: $epoch , g_mse_train: $g_mse_train , g_mse_test: $g_mse_test")
    end

    #@time data_test,positions_test = less_d_faster_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))

    
    g_mse = 0.
    for index in 1:batch_size_create_data
        g_mse += Flux.Losses.mae(model(data_learn[index]),positions[3*(index-1)+1:3*index])/batch_size_create_data
    end
    println()
    println("the average mse is: $g_mse")

    println("the supposed position is: $(positions[1:3]) the prediction is: $(model(data_learn[1]))")

    return train_accuracies,test_accuracies
end


function wild_do_ki_only_times_actual_batches(batch_size_create_data, batches_per_epoch,num_ears ;epochs=2,new_data = 5, print_every = batches_per_epoch//1000, evaluation_batch_size = batch_size_create_data)
    @time data_learn,positions = only_times_and_dist(batch_size_create_data*batches_per_epoch)

    num_layers = 10
    map_to = 50*num_ears
    layers = [Flux.Dense(2*num_ears => map_to) for _ in 1:num_layers]
    model = Flux.Chain(Flux.Parallel(vcat, layers...),
                    Flux.Dense( num_layers*map_to => num_layers*map_to ),
                    Flux.Dense( num_layers*map_to => num_layers*map_to ),
                    Flux.Dense( num_layers*map_to => num_layers ),
                    Flux.Dense( num_layers => 3));


    model = Flux.f64(model)

    opt_state = Flux.setup(Flux.Adam(), model)
    #opt_state = Flux.setup(Flux.Descent(), model)
    #opt_state = Flux.setup(Flux.Momentum(), model)
    #opt_state = Flux.setup(Flux.NADAM(), model)


    train_accuracies = zeros(epochs)
    test_accuracies = zeros(epochs)

    data_test,positions_test = only_times_and_dist(evaluation_batch_size)


    for epoch in 1:epochs

        if epoch%new_data==0
            data_learn,positions = only_times_and_dist(batch_size_create_data*batches_per_epoch)
        end
        
        #model = Flux.trainmode!(model)

        for index in 1:batches_per_epoch
            batch_start = (index - 1) * batch_size_create_data + 1
            batch_end = index * batch_size_create_data
            raw_batch = data_learn[batch_start:batch_end]
            data_batch = reduce(hcat, raw_batch)
            positions_batch = positions[3*(batch_start - 1) + 1 : 3*batch_end]
            positions_batch = reshape(positions_batch, 3, :)
            grads = Flux.gradient(model) do m
                result = m(data_batch)
                Flux.Losses.mse(result, positions_batch)    # L1 pruning now slower
            end
            Flux.update!(opt_state, model, grads[1])

            

            if index%print_every==0
                print("\r")
                print("Epoch: $epoch Advanced: $(round(index/(batches_per_epoch)*100,digits=1))%")
            end
        end

        
        g_mse_train = 0.
        g_mse_test = 0.
        for index in 1:evaluation_batch_size
            g_mse_train += Flux.Losses.mae(model(data_learn[index]),positions[3*(index-1)+1:3*index])/evaluation_batch_size#sqrt(sum((model(data_learn[index]) .- positions[3*(index-1)+1:3*index]).^2))/evaluation_batch_size
            g_mse_test += Flux.Losses.mae(model(data_test[index]),positions_test[3*(index-1)+1:3*index])/evaluation_batch_size#sqrt(sum((model(data_test[index]) .- positions_test[3*(index-1)+1:3*index]).^2))/evaluation_batch_size
        end
        train_accuracies[epoch] = g_mse_train
        test_accuracies[epoch] = g_mse_test
        
        print("\r")
        println("Epoch: $epoch , g_mse_train: $g_mse_train , g_mse_test: $g_mse_test")
    end

    #@time data_test,positions_test = less_d_faster_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))

    
    g_mse = 0.
    for index in 1:batch_size_create_data
        g_mse += Flux.Losses.mae(model(data_learn[index]),positions[3*(index-1)+1:3*index])/batch_size_create_data
    end
    println()
    println("the average mse is: $g_mse")

    println("the supposed position is: $(positions[1:3]) the prediction is: $(model(data_learn[1]))")

    return train_accuracies,test_accuracies
end




saving_plot_path = (@__DIR__)*"/plots/"
saving_data_path = (@__DIR__)*"/data/"
saving_data_to = saving_data_path*"train_batch.txt"

println("HI :)")

batch_size_create_data = 1_000
listening_length = 8
num_ears = 4
epochs = 251
new_data = 2
eval_b_size = 100
batches_per_epoch = 10
print_new_data = batches_per_epoch//100


#@time train_acc,test_acc = do_ki_only_times_actual_batches(batch_size_create_data,batches_per_epoch,num_ears,epochs=epochs ,new_data=new_data,print_every=print_new_data,evaluation_batch_size=eval_b_size)
@time train_acc,test_acc = wild_do_ki_only_times_actual_batches(batch_size_create_data,batches_per_epoch,num_ears,epochs=epochs ,new_data=new_data,print_every=print_new_data,evaluation_batch_size=eval_b_size)


#From here on only plotting
function plot_accuracy(train_acc,test_acc,plot_name)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1],title = "accuracy over epochs",
    xlabel = LaTeXStrings.LaTeXString("epochs"),
    ylabel = LaTeXStrings.LaTeXString("accuracy"))
    Makie.lines!(ax,1:length(train_acc), train_acc,label="train")
    Makie.lines!(ax, 1:length(train_acc),test_acc,label="test")

    Makie.axislegend()
    CairoMakie.display(fig)
    CairoMakie.save(saving_plot_path*plot_name*".png",fig)
end

#println(train_acc,test_acc)

plot_accuracy(train_acc,test_acc,"wild_test")


print("done")

