import fusion
import nervestitcher
import code

data = fusion.load_interest_point_data("./data/a4_ip_0.005.pkl")
match_matrix = fusion.generate_raw_match_data(data[:20], diagonals=1)
code.interact(local=locals())
print(match_matrix)
