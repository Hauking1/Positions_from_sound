import Makie
import LaTeXStrings
import CairoMakie
import LinearAlgebra
import StaticArrays

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

function plot_missmatches(mismatches,plot_name)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1],title = "missmatches amplitude",
    xlabel = LaTeXStrings.LaTeXString("signal_number"),
    ylabel = LaTeXStrings.LaTeXString("mismatch"))
    Makie.scatter!(ax,1:length(mismatches)//2, mismatches[2:2:end],label="times",markersize =5)
    Makie.scatter!(ax,1:length(mismatches)//2, mismatches[1:2:end],label="dists")
    Makie.axislegend()
    CairoMakie.display(fig)
    CairoMakie.save(saving_plot_path*plot_name*".png",fig)
end

function create_batch_signals_full_data(batch_size_create_data::Int, listening_length::Int; mic_rate::Int=44000, dt::Float64=1/mic_rate)
    rand_float_0_1 = rand(Float64,3*batch_size_create_data)
    results = zeros(batch_size_create_data,listening_length)
    for index in range(0,batch_size_create_data-1)
        results[index+1,:] = #=(rand_float_0_1[3*index+1]*90+10)*=#sin.((rand_float_0_1[3*index+2]*20_000+40)* dt *(0:listening_length-1) .+ 2*pi*rand_float_0_1[3*index+3])
    end
    return results
end

function flat_prepare_data_full_learn(data,value_bounds=(0.,100.);speed_sound = 343.,mic_rate::Int=44000,dt::Float64=1/mic_rate,ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0,0),(0,1,0,0),(0,0,1,0))) 
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

function extract_data(data)
    batch_size = length(data)
    listening_length = length(data[1])
    result = Vector{Float64}(undef,2*batch_size)
    for index in 1:batch_size
        row = data[index]
        #println(row)
        result[2*index-1] = maximum(row)
        for index_row in 1:listening_length
            if row[index_row]!=row[index_row+1]
                if row[1]!=0
                    result[2*index] = 0
                    break
                else
                    result[2*index] = index_row
                    break
                end
            end
        end
    end
    return result
end



saving_plot_path = (@__DIR__)*"/plots/"
saving_data_path = (@__DIR__)*"/data/"

listening_length = 4400
batch_size_create_data_viertel = 200
data_learn,compare_to = flat_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data_viertel,listening_length))
extracted = extract_data(data_learn)
plot_missmatches(compare_to .-extracted,"mismatches_direkt_approach")


#plot_single_ear_data(data_learn)

