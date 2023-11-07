from flask import Flask, render_template, request
import pickle
import joblib
import pandas as pd

app = Flask(__name__, template_folder='template')
# Assuming that you have your model training code in a variable named 'clf'
model = pickle.load(open("C:\\Users\\Ganesh\\Desktop\\Miniproject\\flask\\model.pkl", "rb"))


ct = joblib.load("C:\\Users\\Ganesh\\Desktop\\Miniproject\\flask\\feature_values")
gender_mapping = {
    "Male": ["Male", "male", "M", "m", "Male", "Cis Male", "Man", "cis male", "Mail", "Male-ish", "Male (CIS)", "Cis Man", "msle", "Malr", "Mal", "maile", "Make"],
    "Female": ["Female", "female", "F", "f", "Woman", "Female", "femail", "Cis Female", "cis-female/femme", "Femake", "Female (cis)", "woman"],
    "Non-Binary": ["Female (trans)", "queer/she/they", "non-binary", "fluid", "queer", "Androgyne", "Trans-female", "male leaning androgynous", "Agender", "A little about you", "Nah", "All", "ostensibly male, unsure what that really means", "Genderqueer", "Enby", "p", "Neuter", "something kinda male?", "Guy (-ish) ^_^", "Trans Woman"]
}


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/pred')
def predict():
    return render_template("index.html")


@app.route('/out', methods=["POST"])
def output():
    
    age = request.form["age"]
    gender = request.form["gender"]
    self_employed = request.form["self_employed"]
    family_history = request.form["family_history"]
    work_interfere = request.form["work_interfere"]
    no_employees = request.form["no_employees"]
    remote_work = request.form["remote_work"]
    tech_company = request.form["tech_company"]
    benefits = request.form["benefits"]
    care_options = request.form["care_options"]
    wellness_program = request.form["wellness_program"]
    seek_help = request.form["seek_help"]
    anonymity = request.form["anonymity"]
    leave = request.form["leave"]
    mental_health_consequence = request.form["mental_health_consequence"]
    phys_health_consequence = request.form["phys_health_consequence"]
    coworkers = request.form["coworkers"]
    supervisor = request.form["supervisor"]
    mental_health_interview = request.form["mental_health_interview"]
    phys_health_interview = request.form["phys_health_interview"]
    mental_vs_physical = request.form["mental_vs_physical"]
    obs_consequence = request.form["obs_consequence"]
    for mapped_gender, aliases in gender_mapping.items():
        if gender.lower() in [alias.lower() for alias in aliases]:
            gender = mapped_gender
            break

    
    data = [
        [age, gender, self_employed, family_history, work_interfere, no_employees, remote_work, tech_company, benefits,
         care_options, wellness_program, seek_help, anonymity, leave, mental_health_consequence,
         phys_health_consequence, coworkers, supervisor, mental_health_interview, phys_health_interview,
         mental_vs_physical, obs_consequence]]

    feature_cols = ['Age', 'Gender', 'self_employed', 'family_history', 'work_interfere', 'no_employees', 'remote_work',
                    'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
                    'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor',
                    'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence']
    

    pred = model.predict(ct.transform(pd.DataFrame(data, columns=feature_cols)))
    pred = pred[0]

    if pred:
        return render_template("output.html", y="This person requires mental health treatment")
    else:
        return render_template("output.html", y="This person doesn't require mental health treatment")


if __name__ == '__main__':
    app.run(debug=True)