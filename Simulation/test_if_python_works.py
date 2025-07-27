#import julia
#julia.install()

path_scripts = "\\".join(__file__.split("\\")[:-1])+"\\"
#from julia import Main
#Main.include(path_scripts+"making_data.jl")
#data_learn,positions = Main.only_times_and_dist(Main.create_batch_signals_full_data(batch_size_create_data,listening_length))
#Main.println("HI")


from julia import Test_package
import time


batch_size_create_data = 100
listening_length = 8
num_ears = 4
epochs = 50
new_data = 1
print_new_data = batch_size_create_data//100
eval_b_size = 50
start = time.time()
data_learn,positions = Test_package.only_times_and_dist(Test_package.create_batch_signals_full_data(batch_size_create_data,listening_length))
print(f"first call took: {time.time()-start} seconds")
average_time = 0
for _ in range(100):
    start = time.time()
    data_learn,positions = Test_package.only_times_and_dist(Test_package.create_batch_signals_full_data(batch_size_create_data,listening_length))
    average_time += time.time()-start
print(f"Average time is: {average_time/100}")


#print(data_learn)
#print(positions)

print("hi")