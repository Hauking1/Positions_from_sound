import Makie
import LaTeXStrings
import CairoMakie
import Flux
import LinearAlgebra
import StaticArrays
import CUDA


function create_batch_signals_full_data(batch_size_create_data::Int, listening_length::Int; mic_rate::Int=44000, dt::Float64=1/mic_rate)
    rand_float_0_1 = rand(Float64,3*batch_size_create_data)
    results = zeros(batch_size_create_data,listening_length)
    for index in range(0,batch_size_create_data-1)
        results[index+1,:] = rand_float_0_1[3*index+1]*sin.(#=rand_float_0_1[3*index+2]*=# 20_000* dt *(0:listening_length-1)#= .+ 2*pi*rand_float_0_1[3*index+3]=#)
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

function prepare_data_full_learn(data;ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0),(0,1,0),(0,0,1)),speed_sound = 343., mic_rate::Int=44000 , dt::Float64=1/mic_rate)
    batch_size, listening_length = size(data)
    positions_sound = StaticArrays.rand(3*batch_size) .*200 .- 100
    
    results = [Vector{Float64}(undef, 3*listening_length) for _ in 1:batch_size]

    @inbounds for index in range(1,batch_size)
        distances = @StaticArrays.SVector [LinearAlgebra.norm(positions_sound[3*(index-1)+1:3*(index-1)+3] .- ear_positions[n_ear]) for n_ear in 1:3]
        times = ceil.(Int,(distances/speed_sound .- minimum(distances/speed_sound))/dt)

        row = @view data[index, :]
        @inbounds for n_ear in 1:3
            l_index = (n_ear-1)*listening_length+1
            u_index = n_ear*listening_length
            res_write_to = @view results[index][l_index:u_index]
            circshift!(res_write_to,row,times[n_ear])
            @inbounds results[index][l_index:u_index] .*= (1/distances[n_ear]^2)
            @inbounds results[index][l_index:l_index+times[n_ear]] .= 0    #
        end
    end
    return results,permutedims(reshape(positions_sound,(batch_size,3)),[2, 1])
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

function less_d_faster_prepare_data_full_learn(data; ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0,0),(0,1,0,0),(0,0,1,0)), speed_sound = 343., mic_rate::Int=44000, dt::Float64=1/mic_rate, num_ears=length(ear_positions))

    #println("size of ears = $(length(ear_positions))")
    batch_size, listening_length = size(data)
    #positions_sound = StaticArrays.rand(3*batch_size) .*200 .- 100
    #positions_sound = [-1 for _ in 1:3batch_size]
    rand_angles = rand(2*batch_size)*2*pi
    radius = 10
    positions_sound = [[radius*sin(rand_angles[2*index-1])*cos(rand_angles[2*index]), radius*sin(rand_angles[2*index-1])*sin(rand_angles[2*index]), radius*cos(rand_angles[2*index-1])] for index in 1:batch_size]
    positions_sound = [x for y in positions_sound for x in y]
    #println("on creation the position is: $positions_sound")
    
    results = [Vector{Float64}(undef, num_ears*listening_length) for _ in 1:batch_size]

    @inbounds for index in 1:batch_size
        distances = [LinearAlgebra.norm(positions_sound[3*(index-1)+1:3*index] .- ear_positions[n_ear]) for n_ear in 1:num_ears]
        times = ceil.(Int,(distances/speed_sound .- minimum(distances/speed_sound))/dt)
        #println("datacreation uses these positions: $(positions_sound[3*(index-1)+1:3*index])")
        #println(distances)
        #println(times)

        row = @view data[index, :]
        @inbounds for n_ear in 1:num_ears
            l_index = (n_ear-1)*listening_length+1
            u_index = n_ear*listening_length
            res_write_to = @view results[index][l_index:u_index]
            circshift!(res_write_to,row,times[n_ear])
            @inbounds results[index][l_index:u_index] .*= (1/distances[n_ear]^2)
            @inbounds results[index][l_index:l_index+times[n_ear]] .= 0    #
        end
    end
    return results,positions_sound
end

function only_times_and_dist(batch_size; ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0,0),(0,1,0,0),(0,0,1,0)), speed_sound = 343., mic_rate::Int=44000, dt::Float64=1/mic_rate, num_ears=length(ear_positions))
    positions_sound = StaticArrays.rand(3*batch_size) .*200 .- 100
    #positions_sound = [-1 for _ in 1:3batch_size]
    # rand_angles = rand(2*batch_size)*2*pi
    # radius = 10
    # positions_sound = [[radius*sin(rand_angles[2*index-1])*cos(rand_angles[2*index]), radius*sin(rand_angles[2*index-1])*sin(rand_angles[2*index]), radius*cos(rand_angles[2*index-1])] for index in 1:batch_size]
    # positions_sound = [x for y in positions_sound for x in y]
    #println("on creation the position is: $positions_sound")
    
    results = [Vector{Float64}(undef, 2*num_ears) for _ in 1:batch_size]
    #volumes = rand(batch_size)

    @inbounds for index in 1:batch_size
        distances = [LinearAlgebra.norm(positions_sound[3*(index-1)+1:3*index] .- ear_positions[n_ear]) for n_ear in 1:num_ears]
        results[index][1:num_ears] .= [1/(dist^2) for dist in distances] #volumes[index]
        results[index][num_ears+1:end] .= ceil.(Int,(distances/speed_sound .- minimum(distances/speed_sound))/dt)
    end
    return results,positions_sound
end










function do_ki(batch_size_create_data,listening_length,data_learn,positions,num_ears ;epochs=2,new_data = 5, print_every = batch_size_create_data//1000, evaluation_batch_size = batch_size_create_data)
    device = Flux.gpu_device()
    data_learn = data_learn|>device
    positions = positions |>device

    model = Flux.Chain(#Flux.Dense(num_ears*listening_length=>500,Flux.tanhshrink),
        Flux.Dense(num_ears*listening_length=>2*listening_length,Flux.tanhshrink),
        Flux.Dense(2*listening_length=>listening_length,Flux.tanhshrink),
        Flux.Dense(listening_length=>500,Flux.tanhshrink),
        Flux.Dense(500=>500,Flux.tanhshrink),
        Flux.Dense(500=>500,Flux.tanhshrink),
        Flux.Dense(500=>500,Flux.tanhshrink),
        Flux.Dense(500=>500,Flux.tanhshrink),
        Flux.Dense(500=>500,Flux.tanhshrink),
        Flux.Dense(500=>500,Flux.tanhshrink),
        Flux.Dense(500=>500,Flux.tanhshrink),
        Flux.Dense(500=>500,Flux.tanhshrink),
        Flux.Dense(500=>500,Flux.tanhshrink),
        Flux.Dense(500=>500,Flux.tanhshrink),
        Flux.Dense(500=>500,Flux.tanhshrink),
        Flux.Dense(500=>100,Flux.tanhshrink),
        Flux.Dense(100=>12,Flux.tanhshrink),
        Flux.Dense(12=>12,Flux.tanhshrink),
        Flux.Dense(12=>6,Flux.tanhshrink),
        Flux.Dense(6=>3))|>device
    model = Flux.f64(model)

    #opt_state = Flux.setup(Flux.Adam(0.01), model)
    #opt_state = Flux.setup(Flux.Descent(), model)
    opt_state = Flux.setup(Flux.Momentum(), model)


    train_accuracies = zeros(epochs)
    test_accuracies = zeros(epochs)

    data_test,positions_test = less_d_faster_prepare_data_full_learn(create_batch_signals_full_data(evaluation_batch_size,listening_length))|>device
    #println("size of position array: $(size(positions_test))")
    #println("size of data array: $(size(data_learn))")


    for epoch in 1:epochs

        if epoch%new_data==0
            data_learn,positions = less_d_faster_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))|>device
        end
        
        #model = Flux.trainmode!(model)


        for index in 1:batch_size_create_data
            # Calculate the gradient of the objective
            # with respect to the parameters within the model:
            # println()
            # println("modl uses these positions: $(positions[3*(index-1)+1:3*index])")
            # println()
            grads = Flux.gradient(model) do m
                result = m(data_learn[index])
                sqrt(sum((result .- positions[3*(index-1)+1:3*index]).^2)) #Flux.Losses.mse(result, positions[3*(index-1)+1:3*index])#+sum(sum.(abs2,Flux.trainables(m)))    # L1 pruning now slower
            end
            # Update the parameters so as to reduce the objective,
            # according the chosen optimisation rule:
            Flux.update!(opt_state, model, grads[1])
            #print(grads)
            #return
            if index%print_every==0
                print("\r")
                print("Epoch: $epoch Advanced: $(round(index/batch_size_create_data*100,digits=1))%")
            end
        end
        #opt_state = Flux.setup(Flux.Adam(1/(epoch)), model)

        
        g_mse_train = 0.
        g_mse_test = 0.
        for index in 1:evaluation_batch_size
            g_mse_train += sqrt(sum((model(data_learn[index]) .- positions[3*(index-1)+1:3*index]).^2))/evaluation_batch_size#Flux.Losses.mae(model(data_learn[index]),positions[3*(index-1)+1:3*index])/evaluation_batch_size
            g_mse_test += sqrt(sum((model(data_test[index]) .- positions_test[3*(index-1)+1:3*index]).^2))/evaluation_batch_size#Flux.Losses.mae(model(data_test[index]),positions_test[3*(index-1)+1:3*index])/evaluation_batch_size
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

function do_ki_only_times(batch_size_create_data,listening_length,data_learn,positions,num_ears ;epochs=2,new_data = 5, print_every = batch_size_create_data//1000, evaluation_batch_size = batch_size_create_data)
    device = Flux.gpu_device()
    data_learn = data_learn|>device
    positions = positions |>device

    #=model = Flux.Chain(
        Flux.Dense(2*num_ears=>100*num_ears,Flux.tanhshrink),
        Flux.Dense(100*num_ears=>100*num_ears,Flux.tanhshrink),
        Flux.Dense(100*num_ears=>3))|>device =#
    
    #model = Flux.Chain(Flux.Dense(2*num_ears=>3),Flux.tanhshrink)|>device #best model yet
    model = Flux.Chain(Flux.Dense(2*num_ears=>5*num_ears,Flux.tanhshrink),Flux.Dense(5*num_ears=>3))|>device  #with prunin ok
    #model = Flux.Chain(Flux.Dense(2*num_ears=>25*num_ears,Flux.tanhshrink),Flux.Dense(25*num_ears=>25*num_ears,Flux.tanhshrink),Flux.Dense(25*num_ears=>3))|>device



    model = Flux.f64(model)

    #opt_state = Flux.setup(Flux.Adam(), model)
    #opt_state = Flux.setup(Flux.Descent(), model)
    #opt_state = Flux.setup(Flux.Momentum(), model)
    opt_state = Flux.setup(Flux.NADAM(), model)


    train_accuracies = zeros(epochs)
    test_accuracies = zeros(epochs)

    data_test,positions_test = only_times_and_dist(create_batch_signals_full_data(evaluation_batch_size,listening_length))|>device
    #println("size of position array: $(size(positions_test))")
    #println("size of data array: $(size(data_learn))")


    for epoch in 1:epochs

        if epoch%new_data==0
            data_learn,positions = only_times_and_dist(create_batch_signals_full_data(batch_size_create_data,listening_length))|>device
        end
        
        #model = Flux.trainmode!(model)


        for index in 1:batch_size_create_data
            # Calculate the gradient of the objective
            # with respect to the parameters within the model:
            # println()
            # println("modl uses these positions: $(positions[3*(index-1)+1:3*index])")
            # println()
            grads = Flux.gradient(model) do m
                result = m(data_learn[index])
                Flux.Losses.mse(result, positions[3*(index-1)+1:3*index])+sum(sum.(abs2,Flux.trainables(m)))    # L1 pruning now slower
            end
            # Update the parameters so as to reduce the objective,
            # according the chosen optimisation rule:
            Flux.update!(opt_state, model, grads[1])
            #print(grads)
            #return
            if index%print_every==0
                print("\r")
                print("Epoch: $epoch Advanced: $(round(index/batch_size_create_data*100,digits=1))%")
            end
        end
        #opt_state = Flux.setup(Flux.Adam(1/(epoch)), model)

        
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


function do_ki_only_times_actual_batches(batch_size_create_data, batches_per_epoch,listening_length,num_ears ;epochs=2,new_data = 5, print_every = batches_per_epoch//1000, evaluation_batch_size = batch_size_create_data)
    device = Flux.gpu_device()
    @time data_learn,positions = only_times_and_dist(batch_size_create_data*batches_per_epoch)|>device

    model = Flux.Chain(
        Flux.Dense(2*num_ears=>500*num_ears,Flux.tanhshrink),
        Flux.Dense(500*num_ears=>500*num_ears,Flux.tanhshrink),
        Flux.Dense(500*num_ears=>500*num_ears,Flux.tanhshrink),
        Flux.Dense(500*num_ears=>200*num_ears,Flux.tanhshrink),
        Flux.Dense(200*num_ears=>100*num_ears,Flux.tanhshrink),
        Flux.Dense(100*num_ears=>3))|>device
    
    #model = Flux.Chain(Flux.Dense(2*num_ears=>3),Flux.tanhshrink)|>device #best model yet
    #model = Flux.Chain(Flux.Dense(2*num_ears=>5*num_ears,Flux.tanhshrink),Flux.Dense(5*num_ears=>3))|>device  #with prunin ok
    #model = Flux.Chain(Flux.Dense(2*num_ears=>25*num_ears,Flux.tanhshrink),Flux.Dense(25*num_ears=>25*num_ears,Flux.tanhshrink),Flux.Dense(25*num_ears=>3))|>device



    model = Flux.f64(model)

    opt_state = Flux.setup(Flux.Adam(), model)
    #opt_state = Flux.setup(Flux.Descent(), model)
    #opt_state = Flux.setup(Flux.Momentum(), model)
    #opt_state = Flux.setup(Flux.NADAM(), model)


    train_accuracies = zeros(epochs)
    test_accuracies = zeros(epochs)

    data_test,positions_test = only_times_and_dist(evaluation_batch_size)|>device


    for epoch in 1:epochs

        if epoch%new_data==0
            data_learn,positions = only_times_and_dist(batch_size_create_data*batches_per_epoch)|>device
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

batch_size_create_data = 10_000
listening_length = 8
num_ears = 4
epochs = 251
new_data = 25
eval_b_size = 100
batches_per_epoch = 100
print_new_data = batches_per_epoch//100


#@time train_acc,test_acc = do_ki_only_times(batch_size_create_data,listening_length,data_learn,positions,num_ears,epochs=epochs ,new_data=new_data,print_every=print_new_data,evaluation_batch_size=eval_b_size)
@time train_acc,test_acc = do_ki_only_times_actual_batches(batch_size_create_data,batches_per_epoch,listening_length,num_ears,epochs=epochs ,new_data=new_data,print_every=print_new_data,evaluation_batch_size=eval_b_size)

#@time train_acc,test_acc = do_ki(batch_size_create_data,listening_length,data_learn,positions,num_ears,epochs=epochs ,new_data=new_data,print_every=print_new_data,evaluation_batch_size=eval_b_size)


#From here on only plotting
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
    CairoMakie.save(saving_plot_path*"Test_less_d_if_right.png",fig)
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

plot_accuracy(train_acc,test_acc,"only_times_dist_ADAM_all_pos_batches")

#plot_concatenated_dat(data_learn,1)

#plot_single_ear_data(data_learn)



print("done")

