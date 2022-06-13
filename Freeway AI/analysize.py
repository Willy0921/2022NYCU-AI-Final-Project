import os

if __name__ == "__main__":
    compares = ["episode", "rewards_ratio", "traintimesXepisode"]
    print("")
    print("Test 10 times per subject")
    print("")
    for compare in compares:
        files = []
        avgs = []
        path = "./Test_results/DQN/" + compare + "/"
        for filename in os.listdir(path):
            f = open(path + filename, 'r')
            lines = f.readlines()
            files.append(lines[0])
            avgs.append(lines[2])

        if compare == "traintimesXepisode":
            print("")
            print(f"(#train,")
            print(f"#episode)\tDQN")
        elif compare == "rewards_ratio":
            print("")
            print(f"rewards")
            print(f"ratio\tDQN")
        else:
            print(f"{compare}\tDQN")
        print("-----------------------------------------")
        for i in range(len(files)):
            if compare == "traintimesXepisode":
                print(
                    f"({files[i][-6]}, {files[i][-4:-1]})\t{avgs[i]}")
            elif compare == "rewards_ratio":
                print(f"{files[i][18:-1]}\t{avgs[i]}")
            elif compare == "episode":
                print(f"{files[i][12:-1]}\t{avgs[i]}")
    print("")
