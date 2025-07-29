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

function faster_prepare_data_full_learn(data;
    ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0,0),(0,1,0,0),(0,0,1,0)),
    speed_sound = 343.,
    mic_rate::Int=44000,
    dt::Float64=1/mic_rate,
    num_ears::Int=length(ear_positions))    
    
    batch_size, listening_length = size(data)
    positions_sound = rand(3*batch_size)*200 .- 100

    @inbounds distances = [[LinearAlgebra.norm(positions_sound[3*index+1:3*index+3] .- ear_positions[n_ear]) for n_ear in 1:num_ears] for index in range(0,batch_size-1)]
    times = [ceil.(Int,(dist/speed_sound .- minimum(dist/speed_sound))/dt) for dist in distances]
    
    results = [[Vector{Float64}(undef, listening_length) for _ in 1:num_ears] for _ in 1:batch_size]
    @inbounds for index in range(1,batch_size)
        row = @view data[index, :]
        @inbounds for n_ear in 1:num_ears
            circshift!(results[index][n_ear],row,times[index][n_ear])
            results[index][n_ear] .*= (1/distances[index][n_ear]^2)
            @inbounds results[index][n_ear][1:times[index][n_ear]] .= 0
        end
    end
    return results,positions_sound
end

function less_d_faster_prepare_data_full_learn(data;
    ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0,0),(0,1,0,0),(0,0,1,0)),
    speed_sound = 343.,
    mic_rate::Int=44000,
    dt::Float64=1/mic_rate,
    num_ears::Int=length(ear_positions))


    #println("size of ears = $(length(ear_positions))")
    batch_size, listening_length = size(data)
    positions_sound = StaticArrays.rand(3*batch_size) .*20 .- 10
    #positions_sound = [1 for _ in 1:3batch_size]
    #println("on creation the position is: $positions_sound")
    
    results = [Vector{Float64}(undef, num_ears*listening_length) for _ in 1:batch_size]

    @inbounds for index in 1:batch_size
        distances = [LinearAlgebra.norm(positions_sound[3*(index-1)+1:3*index] .- ear_positions[n_ear]) for n_ear in 1:num_ears]
        times = ceil.(Int,(distances/speed_sound .- minimum(distances/speed_sound))/dt)
        #println("datacreation uses these positions: $(positions_sound[3*(index-1)+1:3*index])")
        #println(distances)

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


function ineff_nested_vec_to_3darray(nested_vec,dims)
    # very inefficient conversion of nested vectors into 3d array
    N,M,L = dims
    res = Array{Float64, 3}(undef, N, M, L)

    for i in 1:N
        for j in 1:M
            for k in 1:L
                res[i,j,k] = nested_vec[i][j][k]
            end
        end
    end

    return res
end

function  ineff_nested_vec_to_4darray(nested_vec,dims)
    
    res = Array{Float64, 4}(undef, dims...)
    for i in 1:dims[1]
        for j in 1:dims[2]
            for k in 1:dims[3]
                for l in 1:dims[4]
                    res[i,j,k,l] = nested_vec[i][j][k][l]
                end
            end
        end
    end
    return res
end


function prepare_for_CNN(data,batch_size_create_data,listening_length;num_ears=4)
    res = ineff_nested_vec_to_3darray(data, (batch_size_create_data, num_ears, listening_length))
    res4d = Array{Float64, 4}(undef, size(res)..., 1)
    res4d[:,:,:,1] .= res
    res4d = permutedims(res4d,(2, 3, 4, 1))
    return res4d
end

saving_plot_path = (@__DIR__)*"/plots/"
saving_data_path = (@__DIR__)*"/data/"
saving_data_to = saving_data_path*"train_batch.txt"

println("HI :)")

batch_size_create_data = 1_00
listening_length = 5_00

#@time wrapper_create_save_data_full_data(batch_size_create_data,listening_length,saving_data_to)

@time data_learn,positions = faster_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))

@time data_learn = prepare_for_CNN(data_learn,batch_size_create_data,listening_length) # (listening_length, 3, 1, batch_size_create_data))
modelCNN = Flux.Chain(
# First convolution, operating upon a listeninglengthx3 image
Flux.Conv((1, 200), 1=>16, pad=Flux.SamePad(), Flux.leakyrelu),
Flux.MaxPool((1,3)),
# x -> Flux.maxpool(x, (1,3)),

# Second convolution, operating upon a ...x... image
Flux.Conv((3, 3), 16=>32, pad=Flux.SamePad(), Flux.leakyrelu),
Flux.MaxPool((1,3)),
# x -> Flux.maxpool(x, (1,3)),

# Third convolution, operating upon a ...x... image
Flux.Conv((3, 3), 32=>32, pad=Flux.SamePad(), Flux.leakyrelu),
Flux.MaxPool((1,3)),
# x -> Flux.maxpool(x, (1,3)),

# Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
# which is where we get the 288 in the `Dense` layer below:
Flux.flatten,
# x -> Flux.reshape(x, :, size(x, 4)),
Flux.Dense(Int(floor(listening_length/27)*32*4), 3),

Flux.tanhshrink,
)
modelCNN = Flux.f64(modelCNN)


# @time data_learn,positions = less_d_faster_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))

#@time data_learn,positions = prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))


function do_ki_CNN(batch_size_create_data,listening_length,data_learn,positions,num_ears;epochs=2,new_data = 5, print_every = batch_size_create_data//1000, evaluation_batch_size = batch_size_create_data)
    modelCNN = Flux.Chain(
    # First convolution, operating upon a listeninglengthxnum_ears image
    Flux.Conv((1, 200), 1=>16, pad=Flux.SamePad(), Flux.leakyrelu),
    Flux.MaxPool((1,3)),
    # x -> Flux.maxpool(x, (1,3)),

    # Second convolution, operating upon a ...x... image
    Flux.Conv((num_ears, 3), 16=>32, pad=Flux.SamePad(), Flux.leakyrelu),
    Flux.MaxPool((1,3)),
    # x -> Flux.maxpool(x, (1,3)),

    # Third convolution, operating upon a ...x... image
    Flux.Conv((num_ears, 3), 32=>32, pad=Flux.SamePad(), Flux.leakyrelu),
    Flux.MaxPool((1,3)),
    # x -> Flux.maxpool(x, (1,3)),

    # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
    # which is where we get the 288 in the `Dense` layer below:
    Flux.flatten,
    # x -> Flux.reshape(x, :, size(x, 4)),
    Flux.Dense(Int(floor(listening_length/27)*32*num_ears), 3),
    

    # Flux.tanhshrink,
    Flux.leakyrelu,
    
    )
    model = Flux.f64(modelCNN)

    opt_state = Flux.setup(Flux.Adam(1.), model)
    # opt_state = Flux.setup(Flux.Descent(), model)
    # opt_state = Flux.setup(Flux.Momentum(), model)


    train_accuracies = zeros(epochs)
    test_accuracies = zeros(epochs)

    data_test,positions_test = faster_prepare_data_full_learn(create_batch_signals_full_data(evaluation_batch_size,listening_length))
    data_test = prepare_for_CNN(data_test, evaluation_batch_size, listening_length)
    #println("size of position array: $(size(positions_test))")
    #println("size of data array: $(size(data_learn))")


    for epoch in 1:epochs

        if epoch%new_data==0
            data_learn,positions = faster_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))
            data_learn = prepare_for_CNN(data_learn, batch_size_create_data, listening_length)
        end
        
        model = Flux.trainmode!(model)


        for index in 1:batch_size_create_data
            # Calculate the gradient of the objective
            # with respect to the parameters within the model:
            grads = Flux.gradient(model) do m
                result = m(data_learn[:,:,:,index:index])
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
        # opt_state = Flux.setup(Flux.Adam(10/(2*epoch)), model)

        g_mse_train = 0.
        g_mse_test = 0.
        for index in 1:evaluation_batch_size
            g_mse_train +=Flux.Losses.mse(model(data_learn[:,:,:,index:index]),positions[3*(index-1)+1:3*index])/evaluation_batch_size
            g_mse_test +=Flux.Losses.mse(model(data_test[:,:,:,index:index]),positions_test[3*(index-1)+1:3*index])/evaluation_batch_size
        end
        train_accuracies[epoch] = sqrt(g_mse_train)
        test_accuracies[epoch] = sqrt(g_mse_test)
        print("\r")
        println("Epoch: $epoch , g_mse_train: $(sqrt(g_mse_train)) , g_mse_test: $(sqrt(g_mse_test))")
    end

    #@time data_test,positions_test = less_d_faster_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))

    g_mse = 0.
    for index in 1:batch_size_create_data
        g_mse +=Flux.Losses.mse(model(data_learn[:,:,:,index:index]),positions[3*(index-1)+1:3*index])/batch_size_create_data
    end
    println()
    println("the average mse is: $(sqrt(g_mse))")

    println("the supposed position is: $(positions[1:3]) the prediction is: $(model(data_learn[:,:,:,1:1]))")
    println("ratios: $(model(data_learn[:,:,:,1:1]) ./ positions[1:3])")
    
    # println("mse of model output is: $(Flux.Losses.mae(model(data_learn[:,:,:,1:1]),positions[1:3]))")
    
    return train_accuracies,test_accuracies
end

modelDNN = Flux.Chain(Flux.Dense(3*listening_length=>500,Flux.relu),
    Flux.Dense(500=>100,Flux.relu),
    Flux.Dense(100=>3,Flux.relu))
modelDNN = Flux.f64(modelDNN)


# println((size(modelCNN[1:6](data_learn[:,:,:,1:1]))))
# println(modelCNN(data_learn[:,:,:,1:1]))

@time train_acc,test_acc = do_ki_CNN(batch_size_create_data,listening_length,data_learn,positions,4;epochs=50,new_data=5,print_every=batch_size_create_data//100,evaluation_batch_size=10)



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


plot_accuracy(train_acc,test_acc,"first_cnn_test")


print("done")

