import LinearAlgebra
import StaticArrays


function create_batch_signals_full_data(batch_size_create_data::Int, listening_length::Int; mic_rate::Int=44000, dt::Float64=1/mic_rate)
    rand_float_0_1 = rand(Float64,3*batch_size_create_data)
    results = zeros(batch_size_create_data,listening_length)
    for index in range(0,batch_size_create_data-1)
        results[index+1,:] = #=rand_float_0_1[3*index+1]*=#sin.(#=rand_float_0_1[3*index+2]*=# 20_000* dt *(0:listening_length-1)#= .+ 2*pi*rand_float_0_1[3*index+3]=#)
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

function only_times_and_dist(data; ear_positions=StaticArrays.SVector{3,Float64}.((1,0,0,0),(0,1,0,0),(0,0,1,0)), speed_sound = 343., mic_rate::Int=44000, dt::Float64=1/mic_rate, num_ears=length(ear_positions))
    batch_size, _ = size(data)
    #positions_sound = StaticArrays.rand(3*batch_size) .*200 .- 100
    #positions_sound = [-1 for _ in 1:3batch_size]
    rand_angles = rand(2*batch_size)*2*pi
    radius = 10
    positions_sound = [[radius*sin(rand_angles[2*index-1])*cos(rand_angles[2*index]), radius*sin(rand_angles[2*index-1])*sin(rand_angles[2*index]), radius*cos(rand_angles[2*index-1])] for index in 1:batch_size]
    positions_sound = [x for y in positions_sound for x in y]
    #println("on creation the position is: $positions_sound")
    
    results = [Vector{Float64}(undef, 2*num_ears) for _ in 1:batch_size]

    @inbounds for index in 1:batch_size
        distances = [LinearAlgebra.norm(positions_sound[3*(index-1)+1:3*index] .- ear_positions[n_ear]) for n_ear in 1:num_ears]
        results[index][1:num_ears] .= [1/(dist^2) for dist in distances]
        results[index][num_ears+1:end] .= ceil.(Int,(distances/speed_sound .- minimum(distances/speed_sound))/dt)
    end
    return results,positions_sound
end