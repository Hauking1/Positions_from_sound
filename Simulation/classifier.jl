import Makie
import LaTeXStrings
import CairoMakie
import Flux
import LinearAlgebra
import StaticArrays
#import CUDA
import BSON
import ScikitLearn


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
    target = ceil.(Int,[LinearAlgebra.norm(positions[3*index-2:3*index]) for index in 1:batch_size_create_data*batches_per_epoch])
    clas = ScikitLearn.RandomForestClassifier(50)
    for epoch in 1:epochs
        target_one_hot = zeros(batch_size_create_data,200)
        for index in 1:batch_size_create_data
            target_one_hot[index,target[(epoch-1)*batch_size_create_data+index]]=1
        end
        clas.fit(reshape(data,1,length(data)),target_one_hot)
    end

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
batch_size_create_data = 3
listening_length = 8
num_ears = 4
epochs = 75
new_data = 2
eval_b_size = 100
batches_per_epoch = 1
print_new_data = batches_per_epoch//100

#=
l_bound,u_bound = 2,3
@time positions_plot = only_2d_positions(10000,(l_bound,u_bound))
plot_data(positions_plot,"test_spherical","evaluation_area:[abs($(l_bound)),abs($(u_bound))]")
println("plotted data")
=#

do_classification(batch_size_create_data,batches_per_epoch)

print("done")

