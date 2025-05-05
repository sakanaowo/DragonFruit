def log_to_file(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')
