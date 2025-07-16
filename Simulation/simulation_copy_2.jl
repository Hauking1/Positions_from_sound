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
            results[index][n_ear] *= (1/distances[index][n_ear]^2)
            @inbounds results[index][n_ear][1:times[index][n_ear]] .= 0
        end
    end
    return results,positions_sound
end


function less_d_faster_prepare_data_full_learn(data;ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0),(0,1,0),(0,0,1)),speed_sound = 343., mic_rate::Int=44000 , dt::Float64=1/mic_rate)
    batch_size, listening_length = size(data)
    positions_sound = rand(3*batch_size)*200 .- 100

    @inbounds distances = [@StaticArrays.SVector [LinearAlgebra.norm(positions_sound[3*index+1:3*index+3] .- ear_positions[n_ear]) for n_ear in 1:3] for index in range(0,batch_size-1)]
    times = [ceil.(Int,(dist/speed_sound .- minimum(dist/speed_sound))/dt) for dist in distances]
    
    results = [Vector{Float64}(undef, 3*listening_length) for _ in 1:batch_size]
    for index in range(1,batch_size)    #@inbounds
        row = @view data[index, :]
        for n_ear in 1:3    #@inbounds
            l_index = (n_ear-1)*listening_length+1
            u_index = n_ear*listening_length
            res_write_to = @view results[index][l_index:u_index]
            circshift!(res_write_to,row,times[index][n_ear])
            results[index][l_index:u_index] *= (1/distances[index][n_ear]^2)
            results[index][l_index:l_index+times[index][n_ear]] .= 0    #@inbounds
        end
    end
    return results,positions_sound
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

batch_size_create_data = 1_000
listening_length = 200

#@time wrapper_create_save_data_full_data(batch_size_create_data,listening_length,saving_data_to)

#@time data_learn,positions = faster_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))

@time data_learn,positions = less_d_faster_prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))

#@time data_learn,positions = prepare_data_full_learn(create_batch_signals_full_data(batch_size_create_data,listening_length))

positions = reshape(positions,batch_size_create_data,3) # TODO correctly reshaped or different datapoints mixed?
println(typeof(data_learn))
println(size(positions))

print("done")
#
model = Flux.Chain(Flux.Dense(3*listening_length=>500,Flux.relu),
    # Flux.Dense(2000=>500,Flux.relu),
    Flux.Dense(500=>100,Flux.relu),
    Flux.Dense(100=>3,Flux.relu))


# data =create_batch_signals_full_data(batch_size_create_data,listening_length)

opt_state = Flux.setup(Flux.Adam(0.01), model)

for epoch in 1:10
    Flux.train!(model, [(data_learn[i],positions[i,:]) for i in 1:batch_size_create_data], opt_state) do m, x, y   
        Flux.Losses.mse(m(x),y)
    end
end

println(model(data_learn[1])-positions[1,:])
#

#@time create_save_batch_signal_encoded(batch_size_create_data,saving_data_to)

#print("done")

#=
fig = Makie.Figure()
ax = Makie.Axis(fig[1, 1],title = "many signals",
xlabel = LaTeXStrings.LaTeXString("time"),
ylabel = LaTeXStrings.LaTeXString("signal(t)"))
for index_ear in range(1,1)
    Makie.lines!(ax, data_learn[index_ear],label="signal concatinated: $index_ear")
end
Makie.axislegend()
CairoMakie.display(fig)
CairoMakie.save(saving_plot_path*"Test_less_d.png",fig)
=#

