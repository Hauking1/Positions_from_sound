import Makie
import LaTeXStrings
import CairoMakie
import Flux
import LinearAlgebra
import StaticArrays
import CUDA
import BSON


function spherical_only_times_and_dist(batch_size, value_bounds=(0.,100.); ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0,0),(0,1,0,0),(0,0,1,0)), speed_sound = 343., mic_rate::Int=44000, dt::Float64=1/mic_rate, num_ears=length(ear_positions))
    positions_sound = rand(3*batch_size)
    l_bound = value_bounds[1]
    u_bound = value_bounds[2]
    diff = u_bound-l_bound
    for index in 1:batch_size
        radius = sqrt(positions_sound[3*index-2])*diff+l_bound
        theta = 2*pi*positions_sound[3*index-1]
        phi = 2*pi*positions_sound[3*index]
        positions_sound[3*index-2] = radius*sin(theta)*cos(phi)
        positions_sound[3*index-1] = radius*sin(theta)*sin(phi)
        positions_sound[3*index] = radius*cos(theta)
    end
    
    results = [Vector{Float64}(undef, 2*num_ears) for _ in 1:batch_size]
    #volumes = rand(batch_size)

    @inbounds for index in 1:batch_size
        distances = [LinearAlgebra.norm(positions_sound[3*(index-1)+1:3*index] .- ear_positions[n_ear]) for n_ear in 1:num_ears]
        results[index][1:num_ears] .= [1/(dist^2) for dist in distances] #volumes[index]
        results[index][num_ears+1:end] .= ceil.(Int,(distances/speed_sound .- minimum(distances/speed_sound))/dt)
    end
    return results,positions_sound
end

function spherical_only_2d_positions(batch_size, value_bounds=(0.,100.))
    positions = rand(2*batch_size)
    l_bound = value_bounds[1]
    u_bound = value_bounds[2]
    diff = u_bound-l_bound
    for index in 1:batch_size
        radius = sqrt(positions[2*index-1])*diff+l_bound
        theta = 2*pi*positions[2*index]
        positions[2*index] = radius*sin(theta)
        positions[2*index-1] = radius*cos(theta)
    end
    return positions
end

function only_times_and_dist(batch_size, value_bounds=(0.,200.); ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0,0),(0,1,0,0),(0,0,1,0)), speed_sound = 343., mic_rate::Int=44000, dt::Float64=1/mic_rate, num_ears=length(ear_positions))
    positions_sound = rand(3*batch_size)
    l_bound = value_bounds[1]
    u_bound = value_bounds[2]
    diff = u_bound-l_bound
    mod_value = 1
    for index in 1:batch_size
        value = positions_sound[index]
        if index%(4)==1 #4 weil wir in 3d arbeiten
            if value<0.5
                positions_sound[index] = -2*value*diff-l_bound
            else
                positions_sound[index] = (2*value-1)*diff+l_bound
            end
        else
            positions_sound[index] = value*2*u_bound-u_bound
        end
        if index%9==0
            mod_value+=1
            mod_value%=4
        end
    end
    
    results = [Vector{Float64}(undef, 2*num_ears) for _ in 1:batch_size]
    #volumes = rand(batch_size)

    @inbounds for index in 1:batch_size
        distances = [LinearAlgebra.norm(positions_sound[3*(index-1)+1:3*index] .- ear_positions[n_ear]) for n_ear in 1:num_ears]
        results[index][1:num_ears] .= [1/(dist^2) for dist in distances] #volumes[index]
        results[index][num_ears+1:end] .= ceil.(Int,(distances/speed_sound .- minimum(distances/speed_sound))/dt)
    end
    return results,positions_sound
end

function dimension_only_positions(dimension,batch_size, value_bounds=(0.,200.))
    positions = rand(dimension*batch_size)
    l_bound = value_bounds[1]
    u_bound = value_bounds[2]
    diff = u_bound-l_bound
    #println("diff: $diff")
    mod_value =1
    dim_sqr = dimension^2
    for index in 1:dimension*batch_size
        value = positions[index]
        if index%(dimension+1)==mod_value
            if value<0.5
                positions[index] = -2*value*diff-l_bound
            else
                positions[index] = (2*value-1)*diff+l_bound
            end
        else
            positions[index] = value*2*u_bound-u_bound
        end
        if index%dim_sqr==0
            mod_value+=1
            mod_value%=(dimension+1)
        end
    end
    return positions
end

function give_model_wild_do_ki_only_times_actual_batches(batch_size_create_data, batches_per_epoch,value_bounds = (0.,200.) ;epochs=2,new_data = 5, print_every = batches_per_epoch//1000, evaluation_batch_size = batch_size_create_data, model = nothing)
    device = Flux.gpu_device()
    @time data_learn,positions = spherical_only_times_and_dist(batch_size_create_data*batches_per_epoch,value_bounds)|>device
    data_test,positions_test = spherical_only_times_and_dist(evaluation_batch_size,value_bounds)|>device

    model = model|>device
    opt_state = Flux.setup(Flux.Adam(), model)
    #opt_state = Flux.setup(Flux.NADAM(), model)

    train_accuracies = zeros(epochs)
    test_accuracies = zeros(epochs)

    for epoch in 1:epochs
        if epoch%new_data==0
            data_learn,positions = spherical_only_times_and_dist(batch_size_create_data*batches_per_epoch,value_bounds)|>device
        end

        for index in 1:batches_per_epoch
            batch_start = (index - 1) * batch_size_create_data + 1
            batch_end = index * batch_size_create_data
            raw_batch = data_learn[batch_start:batch_end]
            data_batch = reduce(hcat, raw_batch)
            positions_batch = positions[3*(batch_start - 1) + 1 : 3*batch_end]
            positions_batch = reshape(positions_batch, 3, :)
            grads = Flux.gradient(model) do m
                Flux.Losses.mse(m(data_batch), positions_batch)+ 0.5* mapreduce(p -> sum(abs2, p), +, Flux.trainables(m))
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
        print("Epoch: $epoch , g_mse_train: $g_mse_train , g_mse_test: $g_mse_test")
    end
    
    g_mse = 0.
    for index in 1:batch_size_create_data
        g_mse += Flux.Losses.mae(model(data_learn[index]),positions[3*(index-1)+1:3*index])/batch_size_create_data
    end
    print("\r")
    println()
    println("the average mse is: $g_mse")

    println("the supposed position is: $(positions[1:3]) the prediction is: $(model(data_learn[1]))")

    return train_accuracies,test_accuracies,model
end

function plot_accuracy(train_acc,test_acc,plot_name)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1],title = "accuracy over epochs",
    xlabel = LaTeXStrings.LaTeXString("epochs"),
    ylabel = LaTeXStrings.LaTeXString("accuracy"))
    Makie.lines!(ax,1:length(train_acc), train_acc,label="train")
    Makie.lines!(ax, 1:length(train_acc),test_acc,label="test")

    Makie.axislegend()
    #CairoMakie.display(fig)
    CairoMakie.save(saving_plot_path*plot_name*".png",fig)
end

function plot_data(pos_sound,plot_name,label_text)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1],title = "sound_source_positions",
    xlabel = LaTeXStrings.LaTeXString("x"),
    ylabel = LaTeXStrings.LaTeXString("y"))
    Makie.scatter!(ax,pos_sound[1:2:end], pos_sound[2:2:end],label=label_text)

    Makie.axislegend()
    #CairoMakie.display(fig)
    CairoMakie.save(saving_plot_path*plot_name*".png",fig)
end

function plot_eval_accuracy(test_acc,plot_name)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1],title = "accuracy per model",
    xlabel = LaTeXStrings.LaTeXString("model"),
    ylabel = LaTeXStrings.LaTeXString("mae"))
    Makie.scatter!(ax, 1:length(test_acc),test_acc,label="test",markersize =7)

    Makie.axislegend()
    #CairoMakie.display(fig)
    CairoMakie.save(saving_plot_path*plot_name*".png",fig)
end

function retrain_models()
    io = open(saving_data_path*"end_accuracies.txt", "w")
    io_all = open(saving_data_path*"all_accuracies.txt", "w")

    #plot_data(spherical_only_2d_positions(1000,(float(0),float(100))),"test_full_data","evaluation_area:[abs(0),abs(100)]")


    for model_value in 1:100
        if model_value<6
            @time positions_plot = spherical_only_2d_positions(1000,(float(model_value-1),float(model_value)))
            plot_data(positions_plot,"data_model_$model_value","evaluation_area:[abs($(model_value-1)),abs($(model_value))]")
            println("plotted data")
        end

        model_name = "wild_test_model_$model_value"
        model = BSON.load(saving_data_path*model_name*".bson")[:model]
        model_state = BSON.load(saving_data_path*model_name*"_state"*".bson")[:model_state]
        Flux.loadmodel!(model,model_state)
        println("now training model: $model_value")
        @time train_acc,test_acc,model = give_model_wild_do_ki_only_times_actual_batches(batch_size_create_data,batches_per_epoch,(float(model_value-1),float(model_value)),epochs=epochs ,new_data=new_data,print_every=print_new_data,evaluation_batch_size=eval_b_size,model=model)
        cpu_device = Flux.cpu_device()
        model = model|>cpu_device
        BSON.@save saving_data_path*model_name*"_state"*".bson" model_state=Flux.state(model)
        BSON.@save saving_data_path*model_name*".bson" model
        println("best accuracy in test data was: $(minimum(test_acc))")
        write(io, string(last(train_acc))*","*string(last(test_acc))*"\n")
        write(io_all, join(string.(train_acc),",")*"\t"*join(string.(test_acc),",")*"\n")
        flush(io)
        flush(io_all)

        if model_value%5==1
            plot_accuracy(train_acc,test_acc,"wild_test_$model_value")
        end
    end

    close(io)
    close(io_all)
end

function eval_models(evaluation_batch_size)
    io = open(saving_data_path*"end_accuracies_after_train.txt", "w")
    all_accuracies = Vector{Float64}(undef,100)

    for model_value in 1:100
        model_name = "wild_test_model_$model_value"
        model = BSON.load(saving_data_path*model_name*".bson")[:model]
        model_state = BSON.load(saving_data_path*model_name*"_state"*".bson")[:model_state]
        Flux.loadmodel!(model,model_state)
        println("now evaluating model: $model_value")
        data_test,positions_test = spherical_only_times_and_dist(evaluation_batch_size,(float(model_value-1),float(model_value)))
        data_test = reduce(hcat, data_test)
        positions_test = reshape(positions_test, 3, :)
        test_acc = 0.
        #for index in 1:evaluation_batch_size
        test_acc += Flux.Losses.mae(model(data_test),positions_test)
        #end
        all_accuracies[model_value]=test_acc
        write(io, string(test_acc)*"\n")
        flush(io)
    end

    close(io)
    return all_accuracies
end


saving_plot_path = (@__DIR__)*"/plots_mini_models/"
loading_data_path = (@__DIR__)*"/data/"
saving_data_path = (@__DIR__)*"/mini_models_data/"


println("HI :)")

#model_name = "wild_test_model"
batch_size_create_data = 3_000
listening_length = 8
num_ears = 4
epochs = 10
new_data = 2
eval_b_size = 1000
batches_per_epoch = 10
print_new_data = batches_per_epoch//100


#retrain_models()
all_acc = eval_models(eval_b_size)
plot_eval_accuracy(all_acc,"model_evaluation")

print("done")

