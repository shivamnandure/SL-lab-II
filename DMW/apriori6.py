from itertools import combinations

# -----------------------------
# Step 1: Dataset
# -----------------------------

dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter'],
    ['Milk', 'Butter'],
    ['Bread', 'Milk', 'Butter', 'Eggs'],
    ['Milk', 'Bread'],
    ['Butter', 'Eggs'],
    ['Milk', 'Bread', 'Butter']
]

# -----------------------------
# Step 2: Function to calculate support
# -----------------------------

def get_support(transactions, itemset):

    count = 0

    for transaction in transactions:

        if set(itemset).issubset(set(transaction)):
            count += 1

    return count / len(transactions)

# -----------------------------
# Step 3: Apriori Function
# -----------------------------

def apriori(transactions, min_support):

    # Find unique items
    unique_items = sorted(list(set(item for t in transactions for item in t)))

    # Generate 1-itemsets
    current_candidates = [[item] for item in unique_items]

    all_frequent_itemsets = []

    k = 1

    while current_candidates:

        frequent_itemsets = []

        # Check support
        for itemset in current_candidates:

            support = get_support(transactions, itemset)

            if support >= min_support:
                frequent_itemsets.append((tuple(itemset), support))

        # Stop if no frequent itemsets
        if not frequent_itemsets:
            break

        all_frequent_itemsets.extend(frequent_itemsets)

        # Generate next candidate itemsets
        found_items = sorted(
            list(set(item for itemset, sup in frequent_itemsets for item in itemset))
        )

        current_candidates = list(combinations(found_items, k + 1))

        k += 1

    return all_frequent_itemsets

# -----------------------------
# Step 4: Minimum Support
# -----------------------------

min_support = 0.3

# -----------------------------
# Step 5: Run Apriori
# -----------------------------

frequent_patterns = apriori(dataset, min_support)

# -----------------------------
# Step 6: Display Results
# -----------------------------

print("\nFrequent Itemsets:\n")

for itemset, support in frequent_patterns:
    print("Itemset:", list(itemset), " Support:", round(support, 2))