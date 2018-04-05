from CustomNN import CustomNN

myNet = CustomNN(2,1,2)
row = [1, 0]
res = myNet.forward_propagate(row)
print(res)
