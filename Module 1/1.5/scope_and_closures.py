# 1.5 Scope, Closures, and the LEGB Rule
API_KEY = "sk-test-xxx"  # Global

def get_client():
    base_url = "https://api.anthropic.com"  # Local
    return f"Client({base_url}, key={API_KEY[:6]}...)"

print(get_client())

# Closure: function that remembers enclosing variables
def make_counter(start: int = 0):
    count = [start]  # Mutable enclosure
    def increment():
        count[0] += 1
        return count[0]
    return increment

token_counter = make_counter()
print("Call 1:", token_counter())
print("Call 2:", token_counter())
print("Call 3:", token_counter())
