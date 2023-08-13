import numpy as np

# User-item matrix (rows: users, columns: movies)
# Each row represents a user's ratings for movies (1-5), 0 indicates no rating
user_item_matrix = np.array([
    [5, 4, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 4, 0, 0],
    [4, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 5],
    [0, 0, 0, 0, 0, 4, 4]
])

# Calculate the similarity between users using cosine similarity
def cosine_similarity(user1, user2):
    common_ratings = user_item_matrix[user1] * user_item_matrix[user2]
    norm_user1 = np.linalg.norm(user_item_matrix[user1])
    norm_user2 = np.linalg.norm(user_item_matrix[user2])
    similarity = np.dot(common_ratings, common_ratings) / (norm_user1 * norm_user2)
    return similarity

# Get recommendations for a user using collaborative filtering
def get_recommendations(user):
    similar_users = []
    for other_user in range(len(user_item_matrix)):
        if other_user != user:
            similarity = cosine_similarity(user, other_user)
            similar_users.append((other_user, similarity))
    
    # Sort similar users by descending similarity
    similar_users.sort(key=lambda x: x[1], reverse=True)
    
    # Recommend movies that the user hasn't rated yet
    recommendations = []
    for other_user, similarity in similar_users:
        for movie in range(len(user_item_matrix[user])):
            if user_item_matrix[user][movie] == 0 and user_item_matrix[other_user][movie] > 0:
                recommendations.append(movie)
                if len(recommendations) >= 3:
                    return recommendations

    return recommendations

user_id = 0  # User for whom you want to generate recommendations
recommendations = get_recommendations(user_id)

print("Recommended movies for user", user_id)
for movie_id in recommendations:
    print("Movie", movie_id)
