def dispatch(data, dic):
    return {key:func(data) for key, func in dic.items()}