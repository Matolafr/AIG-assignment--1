using CSV, DataFrames, Plots, Random, MLBase, Statistics
# MLBase is used for a confusion matrix in the end
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
    end
end

# check whether the results vector was assigned correctly (1==yes, 0==no)
(data_matrix[1,end]==0)&&(original[1,end]=="no")


## create two data sets, training and testing set. Also normalize and add the bias

indd = ceil(0.8*size(data_matrix,1))
splitt = convert(Int,indd)

#data_matrix_norm = data_matrix./maximum(data_matrix)

# shuffle the data so that random elements are taken into the training/test set
data_shuffled = data_matrix[shuffle(1:end),:]
x_norm = (data_shuffled[1:end,1:end-1].-mean(data_shuffled[1:end,1:end-1]))./std(data_shuffled[1:end,1:end-1])
train_data = x_norm[1:splitt,1:end]
train_values = data_shuffled[1:splitt,end]
bias = fill(1, splitt)
train_data = hcat(train_data,bias)

test_data = data_shuffled[splitt+1:end,1:end-1]
test_values = data_shuffled[splitt+1:end,end]

## start optimizing

theetaa = apu = fill(0.0, size(train_data)[2])
steps = 200
alphaa = 2
m = size(train_data,1)
lambda = 1
J=0

for i = 1:steps
    apu = (1 .+ exp.(-(train_data*theetaa))).^(-1)
    global theetaa =
    theetaa .- alphaa/m .* (transpose(train_data)*(apu-train_values) .+ lambda.*theetaa)
    #summaus = train_values * transpose(log.(apu)) -
    #(1 .- train_values)*log.(transpose(1 .-apu))
    #J = 1/m*( sum( summaus ) ) + lambda/(2*m)*sum(theetaa[2:end].^2)
end

## predict results based on the fit

prediction = (1 .+ exp.(-(train_data*theetaa))).^-1
prediction_test = (1 .+ exp.(-(test_data*theetaa[1:end-1]))).^-1
# Classify the predictions using a threshold of 0.5
prediction_classified = (prediction.>=0.5)
prediction_test_classified = (prediction_test.>=0.5)
## accuracy calculations

sum(train_values .== prediction_classified)/size(prediction,1)
sum(test_values .== prediction_test_classified)/size(prediction_test,1)

## precision calculations

sum(prediction_classified .== 1 .== train_values)/sum(prediction_classified .== 1)
sum(prediction_test_classified .== 1 .== test_values)/sum(prediction_test_classified .== 1)


## recall calculations

sum(train_values .== 1 .== prediction_classified)/sum(train_values .== 1)
sum(test_values .== 1 .== prediction_test_classified)/sum(test_values .== 1)


## confusion matrices

print(confusmat(2,round.(Int,prediction.+1), round.(Int,train_values.+1)))
