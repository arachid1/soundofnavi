def avg_and_std(audios, logs, name):
    print("yo")
    audios = np.array(audios)
    shape = list(audios.shape)
    shape[0] = 1
    shape = tuple(shape)
    print(shape)
    sum = np.zeros(shape=shape)
    for a in audios:
        print(a.shape)
        sum += a
    avg = sum / len(audios)
    std = np.reshape(np.std(audios, axis=0), newshape=shape)

    print(avg)
    print(avg.shape)
    print(std.shape)

    plt.xlim(-2, 1)
    plt.plot(list(range(0, len(avg))), avg, "x")
    plt.savefig(os.path.join(logs, name + "_avg.png"))

    plt.plot(list(range(0, len(std))), std, "x")
    plt.savefig(os.path.join(logs, name + "_std.png"))

    # avg = Image.fromarray(avg)
    # avg = avg.convert("RGB")
    # avg.save(os.path.join(logs, name + "_avg.png"))
    # std = Image.fromarray(std)
    # std = std.convert("RGB")
    # std.save(os.path.join(logs, name + "_std.png"))
    # pass
