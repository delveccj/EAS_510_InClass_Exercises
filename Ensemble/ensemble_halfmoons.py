from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import time

samples = (500, 5000, 50000, 500000)

for sample in samples:
    print(sample)
    start = time.time()
    X, y = make_moons(n_samples=sample, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('svc', SVC(random_state=42))
        ]
    )
    voting_clf.fit(X_train, y_train)
    end = time.time()

    print(f"Time to train in seconds: {(end - start) / 1000} ")

    # Basic performance
    y_pred = voting_clf.predict(X_test)
    print(f"Voting Classifier Accuracy: {accuracy_score(y_test, y_pred):.3f}")

    # Compare with individual classifiers - use named_estimators_ instead
    print("\nIndividual Classifier Performance:")
    for name in ['lr', 'rf', 'svc']:
        clf = voting_clf.named_estimators_[name]
        individual_pred = clf.predict(X_test)
        print(f"{name.upper()} Accuracy: {accuracy_score(y_test, individual_pred):.3f}")

    # OR alternatively, fit fresh classifiers:
    # lr_clf = LogisticRegression(random_state=42)
    # rf_clf = RandomForestClassifier(random_state=42)
    # svc_clf = SVC(random_state=42)
    # 
    # for name, clf in [('lr', lr_clf), ('rf', rf_clf), ('svc', svc_clf)]:
    #     clf.fit(X_train, y_train)
    #     individual_pred = clf.predict(X_test)
    #     print(f"{name.upper()} Accuracy: {accuracy_score(y_test, individual_pred):.3f}")

    # Detailed classification report
    print("\nVoting Classifier Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Access individual classifier predictions using named_estimators_
    lr_pred = voting_clf.named_estimators_['lr'].predict(X_test)
    rf_pred = voting_clf.named_estimators_['rf'].predict(X_test)
    svc_pred = voting_clf.named_estimators_['svc'].predict(X_test)

    print("\nFirst 10 predictions comparison:")
    print("True labels:", y_test[:10])
    print("Ensemble:   ", y_pred[:10])
    print("LogReg:     ", lr_pred[:10])
    print("RandForest: ", rf_pred[:10])
    print("SVC:        ", svc_pred[:10])

    # See where classifiers disagree
    disagreements = np.where((lr_pred != rf_pred) | (rf_pred != svc_pred) | (lr_pred != svc_pred))[0]
    print(f"\nClassifiers disagree on {len(disagreements)} out of {len(y_test)} samples")
