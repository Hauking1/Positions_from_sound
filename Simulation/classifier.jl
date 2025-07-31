import Makie
import LaTeXStrings
import CairoMakie
import Flux
import LinearAlgebra
import StaticArrays
import JLD2
import DecisionTree


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

function do_classification(batch_size_create_data, batches_per_epoch;epochs=2,new_data = 5, print_every = batches_per_epoch//1000, evaluation_batch_size = batch_size_create_data,)
    data, positions = spherical_only_times_and_dist(batch_size_create_data*batches_per_epoch,(0.,100.))
    data_t, positions_t = spherical_only_times_and_dist(evaluation_batch_size,(0.,100.))
    data = permutedims(reduce(hcat, data),(2,1))
    data_t = permutedims(reduce(hcat, data_t),(2,1))
    #println(data_t)
    #println(positions_t)

    target = ceil.(Int,[LinearAlgebra.norm(positions[3*index-2:3*index]) for index in 1:batch_size_create_data*batches_per_epoch])
    target_t = ceil.(Int,[LinearAlgebra.norm(positions_t[3*index-2:3*index]) for index in 1:evaluation_batch_size])
    #println(target_t)
    model = DecisionTree.RandomForestClassifier()
    train_acc = Float64[]
    test_acc = Float64[]
    #println(size(data))
    #println(size(target))

    DecisionTree.fit!(model,data, target)
    #println(DecisionTree.predict(model, data))

    # Evaluate on train and test sets
    train_acc = sum(DecisionTree.predict(model, data) .== target) / length(target)
    test_acc = sum(DecisionTree.predict(model, data_t) .== target_t) / length(target_t)
    #println(DecisionTree.predict(model, data_t))
    println("Train accuracy: $(round(train_acc; digits=4)), Test accuracy: $(round(test_acc; digits=4))")
    return model, train_acc, test_acc
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




saving_plot_path = (@__DIR__)*"/plots_mini_models/"
loading_data_path = (@__DIR__)*"/data/"
saving_data_path = (@__DIR__)*"/mini_models_data/"


println("HI :)")

#model_name = "wild_test_model"
batch_size_create_data = 100000
listening_length = 8
num_ears = 4
epochs = 75
new_data = 2
eval_b_size = 10000
batches_per_epoch = 1
print_new_data = batches_per_epoch//100


classifier, train_acc, test_acc = do_classification(batch_size_create_data,batches_per_epoch,evaluation_batch_size=eval_b_size)
#println(typeof(classifier))
JLD2.@save saving_data_path*"classifier_decision_tree.jld2" classifier

print("done")

