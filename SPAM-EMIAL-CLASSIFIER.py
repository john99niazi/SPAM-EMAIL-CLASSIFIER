import pandas as pd
import numpy as np

# Sample email dataset in dictionary format
data = {
    "text": [
        "Congratulations! You've won a free gift card. Click here to claim it.",
        "Your account has been compromised. Please reset your password immediately.",
        "Don't miss out on this limited time offer! Shop now and save 50%.",
        "Meeting scheduled for tomorrow at 10 AM. Please confirm your availability.",
        "Reminder: Your invoice is due for payment. Check your account for details.",
        "Get ready for the biggest sale of the season! Exclusive discounts just for you.",
        "Urgent: Update your billing information to avoid service interruption.",
        "Your package has been shipped. Track your order here for updates.",
        "Hey, just wanted to check in and see how you're doing. Let's catch up soon.",
        "Free trial ending soon! Upgrade now to keep your premium benefits.",
        "Action required: Verify your email to activate your account.",
        "It was great seeing you last weekend. Let me know when you're free again.",
        "Congratulations on your new role! Wishing you success in this new chapter.",
        "Limited seats available for our upcoming workshop. Reserve yours today!",
        "Your weekly newsletter is here! Get the latest updates and tips.",
        "Win big! Enter our sweepstakes for a chance to win a dream vacation.",
        "Team meeting agenda for next week has been updated. Please review.",
        "Your subscription is about to expire. Renew now to continue enjoying our services.",
        "Quick question: Are you available for a call later this week?",
        "Last chance to grab these amazing deals before they're gone forever!",
        "You've been selected for a limited-time credit card offer. Apply today!",
        "Hi, just confirming our lunch meeting tomorrow at noon. See you then!",
        "Exclusive sale! Save up to 80% on select items. Offer ends soon!",
        "Your Netflix subscription has been successfully renewed. Thank you!",
        "Win a free vacation! Click here to enter the sweepstakes now.",
        "Reminder: Your doctor appointment is scheduled for next Monday at 3 PM.",
        "Hurry! Final hours to get 30% off all electronics. Shop now!",
        "Important: Please update your contact details to avoid account suspension.",
        "Looking forward to seeing you at the family gathering this weekend!",
        "Claim your free trial for our premium service. Limited spots available!",
        "We've received your payment. Thank you for your continued support!",
        "URGENT: Security alert detected on your account. Verify now.",
        "Exciting news! Your favorite product is back in stock. Don't miss out!",
        "Don't forget: Project submission deadline is this Friday at 5 PM.",
        "Congratulations! You've been pre-approved for a personal loan.",
        "Thank you for attending the webinar. Here's the recording link.",
        "Your password reset request has been processed successfully.",
        "Special deal: Buy one get one free! Offer valid for 24 hours.",
        "Meeting rescheduled to Thursday at 2 PM. Please confirm your availability.",
        "Last reminder: Claim your exclusive discount code before it expires!",
        "Get the latest updates on your favorite sports. Subscribe now for free.",
        "Your account balance is low. Please add funds to continue services.",
        "Exclusive rewards! Earn points on every purchase. Join today!",
        "Meeting postponed to next week. Please check your calendar for updates.",
        "Congratulations! You've been shortlisted for our prize draw. Act fast!",
        "Your delivery is delayed due to unforeseen circumstances. Thank you for your patience.",
        "Double your income with this simple investment trick. Sign up now!",
        "Don't miss our webinar on career growth strategies this Friday.",
        "Special offer just for you! Buy two items, get one free. Limited time only.",
        "Hey! Just checking in to see if you're free this weekend for coffee.",
        "Reminder: Complete your profile to access premium membership benefits.",
        "Hi, your invoice for the month has been attached. Please review.",
        "Win big prizes every week! Click here to participate in our sweepstakes.",
        "Team meeting notes have been updated. Please review before Monday.",
        "Urgent: Your account has been locked due to suspicious login attempts.",
        "Join our fitness challenge and transform your body in 30 days!",
        "Thanks for your purchase! Your receipt has been sent to your email.",
        "Limited offer: Upgrade to premium and get two months free. Subscribe now.",
        "Hi! I found a great book recommendation for you. Let me know what you think.",
        "Your password will expire in 3 days. Update it now to avoid losing access.",
        "Special alert: Your account will be closed if not updated within 24 hours.",
        "Hi there! Can you share the notes from yesterday’s meeting? Thanks.",
        "Win a $500 gift card by answering a simple survey. Don’t miss out!",
        "Your flight itinerary has been confirmed. Have a safe trip!",
        "New opportunity: Earn $1,000 per week working from home. Sign up now!",
        "Hello, just reminding you about your dentist appointment tomorrow.",
        "Flash sale: Get up to 70% off on all items for the next 6 hours.",
        "Thank you for attending our event. Here's a recap and resources for you.",
        "Hurry! Get an exclusive discount on your next purchase. Offer ends soon!",
        "Quick reminder: Please send the updated proposal by the end of the day.",
        "Act now! Double your earnings with this proven investment strategy.",
        "Your subscription renewal failed. Please update your payment information.",
        "Good news! Your loan application has been pre-approved. Apply today.",
        "Can you review the attached document and share your feedback by Friday?",
        "Exclusive access: Join our VIP club and unlock amazing benefits.",
        "Friendly reminder: Your library books are due for return this Thursday.",
        "You’re invited: Join us for a networking event this weekend.",
        "Urgent: Your delivery could not be completed. Please confirm your address.",
        "Thank you for your payment! Your order will be shipped shortly.",
        "Last chance: Enter our sweepstakes to win a luxury car!"
    ],
    "label": [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1,1, 0, 1, 0, 1,
              0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
              1, 0, 1, 0, 1, 1, 0, 1, 0, 1,1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1]  # 1 = Spam, 0 = Not Spam
}

# Convert dictionary to DataFrame
data = pd.DataFrame(data)

# Preprocess emails (tokenization and lowercasing)
def preprocess_text(text):
    return text.lower().split()

# Create a vocabulary and calculate word frequencies per class
vocabulary = {}
word_counts = {0: {}, 1: {}}

for i, row in data.iterrows():
    label = row['label']
    words = preprocess_text(row['text'])
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
        if word not in word_counts[label]:
            word_counts[label][word] = 0
        word_counts[label][word] += 1

# Calculate probabilities for Naive Bayes classification
class_totals = data['label'].value_counts().to_dict()
vocab_size = len(vocabulary)
def calculate_word_probability(word, label):
    word_frequency = word_counts[label].get(word, 0)
    return (word_frequency + 1) / (sum(word_counts[label].values()) + vocab_size)

# Naive Bayes classification
def classify_email(email):
    words = preprocess_text(email)
    spam_score = np.log(class_totals[1] / len(data))
    not_spam_score = np.log(class_totals[0] / len(data))

    for word in words:
        spam_score += np.log(calculate_word_probability(word, 1))
        not_spam_score += np.log(calculate_word_probability(word, 0))

    return 1 if spam_score > not_spam_score else 0

# Evaluate the model
def evaluate_model(data):
    correct = 0
    for i, row in data.iterrows():
        prediction = classify_email(row['text'])
        if prediction == row['label']:
            correct += 1
    return correct / len(data)

accuracy = evaluate_model(data)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Allow user to input new email for testing
new_email = input("Enter a new email to classify: ")
result = classify_email(new_email)
print(f"The email '{new_email}' is classified as: {'Spam' if result == 1 else 'Not Spam'}")