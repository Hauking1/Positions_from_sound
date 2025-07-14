import LinearAlgebra
import Makie
import LaTeXStrings
import CairoMakie


function create_batch_signals(batch_size::Int, listening_length::Int; mic_rate::Int=44000, dt::Float64=1/mic_rate)
    rand_float_0_1 = rand(Float64,3*batch_size)
    results = zeros(batch_size,listening_length)
    for index in range(0,batch_size-1)
        results[index+1,:] = rand_float_0_1[3*index+1]*sin.(rand_float_0_1[3*index+2]* dt *(0:listening_length-1) .+ rand_float_0_1[3*index+3])
    end
    return results
end

function save_signals(saving_data_path::String,signals)
    open(saving_data_path*"train_batch.txt","w") do file
        for index in range(1,batch_size)
            println(file,join(signals[index,:],","))
        end
    end
end

function wrapper_create_save_data(batch_size::Int,listening_length::Int,saving_data_path::String)
    save_signals(saving_data_path,create_batch_signals(batch_size,listening_length))
end



saving_plot_path = (@__DIR__)*"/plots/"
saving_data_path = (@__DIR__)*"/data/"

println("HI :)")

batch_size = 1000
listening_length = 44_0

@time wrapper_create_data(batch_size,listening_length,saving_data_path)


#=
fig = Makie.Figure()
ax = Makie.Axis(fig[1, 1],title = "many signals",
xlabel = LaTeXStrings.LaTeXString("time"),
ylabel = LaTeXStrings.LaTeXString("signal(t)"))
for index_ear in range(1,3)
    Makie.lines!(ax,range(0.,1.,length = length(signals[1,:])), signals[index_ear,:],label="signal_ear: $index_ear")
end
Makie.axislegend()
CairoMakie.display(fig)
CairoMakie.save(saving_plot_path*"Test.png",fig)
=#
