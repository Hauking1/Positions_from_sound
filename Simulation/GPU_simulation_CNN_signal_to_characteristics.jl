import Makie
import LaTeXStrings
import CairoMakie
import Flux
import JLD2
import LinearAlgebra
import StaticArrays


function create_batch_signals_full_data(batch_size_create_data::Int, listening_length::Int; mic_rate::Int=44000, dt::Float64=1/mic_rate)
    rand_float_0_1 = rand(Float64,3*batch_size_create_data)
    results = zeros(batch_size_create_data,listening_length)
    for index in range(0,batch_size_create_data-1)
        results[index+1,:] = #=(rand_float_0_1[3*index+1]*90+10)*=#sin.((rand_float_0_1[3*index+2]*20_000+40)* dt *(0:listening_length-1) .+ 2*pi*rand_float_0_1[3*index+3])
    end
    return results
end


function faster_prepare_data_full_learn(data,value_bounds=(0.,100.);
    ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0,0),(0,1,0,0),(0,0,1,0)),
    speed_sound = 343.,
    mic_rate::Int=44000,
    dt::Float64=1/mic_rate,
    num_ears::Int=length(ear_positions))    
    
    batch_size, listening_length = size(data)
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

    distances = [[LinearAlgebra.norm(positions_sound[3*index+1:3*index+3] .- ear_positions[n_ear]) for n_ear in 1:num_ears] for index in range(0,batch_size-1)]
    times = [ceil.(Int,(dist/speed_sound .- minimum(dist/speed_sound))/dt) for dist in distances]
    
    results = [[Vector{Float64}(undef, listening_length) for _ in 1:num_ears] for _ in 1:batch_size]
    for index in range(1,batch_size)
        row = @view data[index, :]
        for n_ear in 1:num_ears
            circshift!(results[index][n_ear],row,times[index][n_ear])
            results[index][n_ear] .*= (1/distances[index][n_ear]^2)
            results[index][n_ear][1:times[index][n_ear]] .= 0
        end
    end
    return results,times,distances
end

function flat_prepare_data_full_learn(data,value_bounds=(0.,100.);
    speed_sound = 343.,
    mic_rate::Int=44000,
    dt::Float64=1/mic_rate,
    ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0,0),(0,1,0,0),(0,0,1,0))
    ) 
    num_ears = length(ear_positions)  
    
    batch_size, listening_length = size(data)
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

    disttime = Vector{Float64}(undef, num_ears*batch_size*2)
    results = [Vector{Float64}(undef, listening_length) for _ in 1:num_ears*batch_size]
    @inbounds for index in 1:batch_size
        distances = [LinearAlgebra.norm(positions_sound[3*(index-1)+1:3*index] .- ear_positions[n_ear]) for n_ear in 1:num_ears]
        times = ceil.(Int,(distances/speed_sound .- minimum(distances/speed_sound))/dt)
        row = @view data[index, :]

        @inbounds for n_ear in 1:num_ears
            result_idx = (index - 1)*num_ears + n_ear
            #res_write_to = @view results[num_ears*index+1-(num_ears-(n_ear-1))]
            circshift!(results[result_idx],row,times[n_ear])
            @inbounds results[result_idx] .*= (1/distances[n_ear]^2)
            @inbounds results[result_idx][1:times[n_ear]] .= 0
            disttime[2*result_idx-1]= 1/(distances[n_ear]^2)
            disttime[2*result_idx]= times[n_ear]
        end
    end
    return results,disttime
end

function flat_do_ki_CNN(batch_size_create_data,listening_length,batches_per_epoch,num_ears,saving_data_path;epochs=2,new_data = 5, print_every = batches_per_epoch//100, evaluation_batch_size = batch_size_create_data,load_model=true)
    modelCNN = Flux.Chain(
    # First convolution, operating upon a listeninglengthxnum_ears image
    Flux.Conv((1,200), 1=>8, pad=Flux.SamePad(), Flux.leakyrelu),
    Flux.MaxPool((1,3)),

    # Second convolution, operating upon a ...x... image
    Flux.Conv((1,100), 8=>16, pad=Flux.SamePad(), Flux.leakyrelu),
    Flux.MaxPool((1,3)),

    # Third convolution, operating upon a ...x... image
    Flux.Conv((1,50), 16=>16, pad=Flux.SamePad(), Flux.leakyrelu),
    Flux.MaxPool((1,3)),
    Flux.Conv((1,10), 16=>16, pad=Flux.SamePad(), Flux.leakyrelu),
    Flux.MaxPool((1,3)),

    # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
    # which is where we get the 288 in the `Dense` layer below:
    Flux.flatten,
    Flux.Dense(Int(floor(listening_length/(27*3))*16)=>100,Flux.relu),
    Flux.Dense(100=>2,Flux.relu))

    model = Flux.f64(modelCNN)
    println(Flux.display(model))

    if load_model == true
        model_state = JLD2.load(saving_data_path*"CNN_single_signal_gpu.jld2", "model_state")
        Flux.loadmodel!(model, model_state)
    end

    # opt_state = Flux.setup(Flux.NADAM(), model)
    opt_state = Flux.setup(Flux.Adam(), model)
    # opt_state = Flux.setup(Flux.Descent(), model)
    # opt_state = Flux.setup(Flux.Momentum(), model)


    train_accuracies = zeros(epochs)
    test_accuracies = zeros(epochs)

    X_train,Y_train = flat_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data*batches_per_epoch,listening_length))

    X_test,Y_test = flat_prepare_data_full_learn(create_batch_signals_full_data(evaluation_batch_size,listening_length))


    for epoch in 1:epochs

        if epoch%new_data==0
            X_train,Y_train = flat_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data*batches_per_epoch,listening_length))
        end
        
        #model = Flux.trainmode!(model)


        for index in 1:batches_per_epoch
            batch_start = (index-1) * batch_size_create_data + 1
            batch_end = index * batch_size_create_data
            rawX_batch = X_train[batch_start:batch_end]
            X_batch = reshape(reduce(hcat, rawX_batch), 1, listening_length, 1, batch_size_create_data)
            rawY_batch = Y_train[2*batch_start-1:2*batch_end]
            Y_batch = reshape(rawY_batch, 2, :)

            # Calculate the gradient of the objective
            # with respect to the parameters within the model:
            grads = Flux.gradient(model) do m
                result = m(X_batch)
                Flux.Losses.mse(result, Y_batch) + sum(sum.(abs2,Flux.trainables(m)))    # L1 pruning now slower
            end
            # Update the parameters so as to reduce the objective,
            # according the chosen optimisation rule:
            Flux.update!(opt_state, model, grads[1])
            #print(grads)
            #return
            if index%print_every==0
                print("\r")
                print("Epoch: $epoch Advanced: $(round(index/batches_per_epoch*100,digits=1))%")
            end
        end
        # opt_state = Flux.setup(Flux.Adam(10/(2*epoch)), model)

        g_mse_train = 0.
        g_mse_test = 0.
        for index in 1:evaluation_batch_size
            g_mse_train +=Flux.Losses.mse(model(reshape(X_train[index],1,listening_length,1,1)),Y_train[2*index-1:2*index])/evaluation_batch_size
            g_mse_test +=Flux.Losses.mse(model(reshape(X_test[index],1,listening_length,1,1)),Y_test[2*index-1:2*index])/evaluation_batch_size
        end
        train_accuracies[epoch] = sqrt(g_mse_train)
        test_accuracies[epoch] = sqrt(g_mse_test)
        print("\r")
        println("Epoch: $epoch , g_mse_train: $(sqrt(g_mse_train)) , g_mse_test: $(sqrt(g_mse_test))")
    end

    g_mse = 0.
    for index in 1:batch_size_create_data
        g_mse +=Flux.Losses.mse(model(reshape(X_train[index],1,listening_length,1,1)),Y_train[2*index-1:2*index])/batch_size_create_data
    end
    println()
    println("the average mse is: $(sqrt(g_mse))")

    println("the supposed time, amplitude is: $(Y_train[1:2]) the prediction is: $(model(reshape(X_train[1],1,listening_length,1,1)))")
    
    # println("mse of model output is: $(Flux.Losses.mae(model(X_train[:,:,:,1:1]),positions[1:3]))")

    # save trained model
    device_cpu = Flux.cpu_device()
    model = model|>device_cpu
    model_state = Flux.state(model)
    JLD2.jldsave(saving_data_path*"CNN_single_signal_gpu.jld2"; model_state)
    
    return train_accuracies,test_accuracies,model
end

saving_plot_path = (@__DIR__)*"/plots/"
saving_data_path = (@__DIR__)*"/data/"
saving_data_to = saving_data_path*"train_batch.txt"

listening_length = 4400
batch_size_create_data_viertel = 200
batches_per_epoch = 10
num_ears = 4
epochs = 50
new_data = 2
print_ever = 1
evaluation_batch_size = 100
load_model = true

@time train_acc,test_acc,model = flat_do_ki_CNN(batch_size_create_data_viertel,listening_length,batches_per_epoch,num_ears,saving_data_path;epochs=epochs,new_data=new_data,print_every=print_ever,evaluation_batch_size=evaluation_batch_size,load_model=load_model)


function plot_single_ear_data(data_learn;num_ears=4)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1],title = "many signals",
    xlabel = LaTeXStrings.LaTeXString("time/dt"),
    ylabel = LaTeXStrings.LaTeXString("signal(t)"))
    for index_ear in range(1,num_ears)
        Makie.lines!(ax, data_learn[index_ear],label="signal ear: $index_ear")
    end
    # Makie.lines!(collect(Base.Iterators.flatten(data_learn))[1],label="test")
    Makie.axislegend()
    CairoMakie.display(fig)
    CairoMakie.save(saving_plot_path*"Test_ears_gpu_test.png",fig)
end

function plot_accuracy(train_acc,test_acc,plot_name)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1],title = "loss over epochs",
    xlabel = LaTeXStrings.LaTeXString("epochs"),
    ylabel = LaTeXStrings.LaTeXString("loss"))
    Makie.lines!(ax,1:length(train_acc), train_acc,label="train")
    Makie.lines!(ax, 1:length(train_acc),test_acc,label="test")

    Makie.axislegend()
    CairoMakie.display(fig)
    CairoMakie.save(saving_plot_path*plot_name*".png",fig)
end

#data_learn,_ = flat_prepare_data_full_learn(create_batch_signals_full_data(4,listening_length))
#plot_single_ear_data(data_learn)

plot_accuracy(train_acc,test_acc,"first_cnn_test_single_signal_2_gpu")