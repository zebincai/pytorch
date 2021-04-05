
def max_sub_array(data):
    if len(data) == 0:
        return 0
    max_num = 0
    last_val = data[0]
    sum = last_val
    for idx in range(1, len(data)):
        if data[idx] == last_val:

            sum += data[idx]
            print("line 12 :{}".format(sum))
        else:
            if sum > max_num:
                max_num = sum
            print(idx)
            print("line 17 :{}".format(data[idx]))
            sum = data[idx]
            last_val = data[idx]
    print(sum)
    print(max_num)
    if sum > max_num:
        max_num = sum

    return max_num


if __name__ == "__main__":
    data = [0, 1, 0, 0, 1, 1, 1, 0, 0]
    data = [1, 2, 3, 4, 5]
    data =  [1, 2, 3, 4, 4, 5]
    out = max_sub_array(data)
    print(out)