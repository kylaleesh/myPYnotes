#INSERTION SORT

# 1
pairs = [(5, "apple"), (2, "banana"), (9, "cherry")]
# 2
# pairs = (3, "cat"), (3, "bird"), (2, "dog")]


res = []
res.append(pairs.copy())
for i in range(1,len(pairs)):
    x = pairs[i] 
    for j in range(i-1,-1,-1):
        if pairs[j][0]> x[0]:
            pairs[j+1] = pairs[j]
            pairs[j] = x
    res.append(pairs.copy())
print(res)

'''
# Output1:
[[(5, "apple"), (2, "banana"), (9, "cherry")], 
 [(2, "banana"), (5, "apple"), (9, "cherry")], 
 [(2, "banana"), (5, "apple"), (9, "cherry")]]
# Output2:
[[(3, "cat"), (3, "bird"), (2, "dog")], 
 [(3, "cat"), (3, "bird"), (2, "dog")],
 [(2, "dog"), (3, "cat"), (3, "bird")]]
'''

