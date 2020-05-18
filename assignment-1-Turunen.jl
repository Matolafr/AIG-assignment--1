using CSV, DataFrames, Plots, Random, MLBase

# Import the data, convert to matrix form and leave unnecessary data out

original = CSV.read("/home/matti/kurssit/aig/bank-additional-full.csv")
#old = [1,3,5,7,9,11,13,15,16,19,21]
data_matrix = convert(Matrix,original[:,:])

## preprocess the data, turn categorical data into numbers
#yy = [2,3,4,5,8,11]
for i = [2,3,4,5,6,7,8,9,10,15,21]
    uniquee = unique(data_matrix[:,i])
    for j = 1:size(data_matrix)[1]
        for k = 1:size(uniquee)[1]
            if data_matrix[j,i] == uniquee[k]
                data_matrix[j,i] = Int(k) - 1
            end
        end
#        if i == 21
#            if data_matrix[j,i] == "yes"
#                data_matrix[j,i] = 1
#            else data_matrix[j,i] = 0
#            end
#        end
    end
    #println(i)
end

## create two data sets, training and testing set. Also normalize and add the bias

indd = ceil(0.8*size(data_matrix,1))
splitt = convert(Int,indd)

data_shuffled = data_matrix[shuffle(1:end),:]
train_data = data_shuffled[1:splitt,1:end-2]
train_values = data_shuffled[1:splitt,end]
bias = fill(1, splitt)
train_data = hcat(train_data,bias)
#train_data = train_data./maximum(train_data)

test_data = data_shuffled[splitt+1:end,1:end-2]
test_values = data_shuffled[splitt+1:end,end]


## start optimizing

theetaa = apu = fill(0.0, size(train_data)[2])
steps = 1000
alphaa = 1
m = size(train_data,1)
lambda = 1

for i = 1:steps
    apu = (1 .+ exp.(-(train_data*theetaa))).^-1 - train_values
    global theetaa =
    theetaa .- alphaa/m .* (transpose(train_data)*(apu) .+ lambda.*theetaa)
end


## predict results based on the fit

prediction = (1 .+ exp.(-(train_data*theetaa))).^-1
prediction_test = (1 .+ exp.(-(test_data*theetaa[1:end-1]))).^-1

## accuracy calculations

sum(train_values .== prediction)/size(prediction,1)
sum(test_values .== prediction_test)/size(prediction_test,1)


## precision calculations

sum(prediction .== 1 .== train_values)/sum(prediction .== 1)
sum(prediction_test .== 1 .== test_values)/sum(prediction_test .== 1)


## recall calculations

sum(train_values .== 1 .== prediction)/sum(train_values .== 1)
sum(test_values .== 1 .== prediction_test)/sum(test_values .== 1)


## confusion matrices

print(confusmat(2,round.(Int,prediction.+1), round.(Int,train_values.+1)))
