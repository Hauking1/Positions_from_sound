import Makie
import LaTeXStrings
import CairoMakie
import LinearAlgebra
import StaticArrays
import NLsolve
import JLD2
import BSON
import DecisionTree

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
    return results,disttime,positions_sound
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

function solve_system(d1::Real, d2::Real, d3::Real, d4::Real;x0::Real=50., y0::Real=10., z0::Real=5.,tol::Real=1e-12)

    function F!(F, vars)
        x, y, z = vars
        num = (1 - x)^2 + y^2 + z^2

        F[1] =(d1^2 / d2^2) - (x^2 + (1 - y)^2 + z^2)/num
        F[2] =(d1^2 / d3^2) - (x^2 + y^2 + (1 - z)^2)/num
        F[3] = (d1^2 / d4^2) -(x^2 + y^2 + z^2)/num
    end

    initial = [x0, y0, z0]
    sol = NLsolve.nlsolve(F!, initial; xtol=tol, ftol=tol)
    #println(sol)

    if NLsolve.converged(sol)
        return -1 * sol.zero
    else
        error("Keine Konvergenz: ", sol)
    end
end

saving_plot_path = (@__DIR__)*"/plots/"
saving_data_path = (@__DIR__)*"/data/"
saving_mini_data_path = (@__DIR__)*"/mini_models_data/"

listening_length = 4400
number_data_point = 10

full_signals,compare_to,pos_sound = flat_prepare_data_full_learn(create_batch_signals_full_data(number_data_point,listening_length))
extracted = extract_data(full_signals)
for index in 1:number_data_point
    extracted[8*(index-1)+1:8*index] .= vcat(extracted[8*(index-1)+1:2:8*index],extracted[8*(index-1)+2:2:8*index])
end
JLD2.@load saving_mini_data_path*"classifier_decision_tree.jld2" classifier
models = Vector{Int}(undef,number_data_point)
#println(extracted)
dat_class = reshape(extracted,8,Int(length(extracted)//8))'
#println(dat_class)
#println(DecisionTree.predict(classifier, dat_class))
models .= Int.(DecisionTree.predict(classifier, dat_class))
#println(pos_sound)
norms = [LinearAlgebra.norm(pos_sound[3*(index-1)+1:3*index]) for index in 1:number_data_point]
println("norm of position: $(norms)")
println("models to be chosen: $models")
positions_predicted = Vector{Float64}(undef,3*number_data_point)
for index in 1:number_data_point
    model_name = "wild_test_model_$(models[index])"
    model_dnn = BSON.load(saving_mini_data_path*model_name*".bson")[:model]
    model_state = BSON.load(saving_mini_data_path*model_name*"_state"*".bson")[:model_state]
    Flux.loadmodel!(model_dnn,model_state)
    positions_predicted[3*(index-1)+1:3*index] .= model_dnn(extracted[8*(index-1)+1:8*index])
end
println("predicted positions: $(positions_predicted)")
println("actual positions: $pos_sound")
#plot_missmatches(compare_to .-extracted,"mismatches_direkt_approach")

#following for analytical solution

dists_extract = extracted[1:2:end]
dists_extract = [sqrt(1/test) for test in dists_extract]
positions_ana = Vector{Float64}(undef,3*number_data_point)
for index in 1:batch_size_create_data_viertel
    positions_ana[3*(index-1)+1:3*index] = solve_system(dists_extract[4*(index-1)+1:4*index]...)
end
println("the \"analytical\" solution: $positions_ana")

