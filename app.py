from flask import Flask
from flask import render_template
from flask import request
from flask import redirect,url_for,session, make_response
import mysql.connector
import os
import numpy as np
import pandas as pd
from numpy import linalg
from scipy.optimize import minimize
import math
import random
from scipy.stats import kurtosis 
import scipy.stats as st




app = Flask(__name__)
app.secret_key=os.urandom(24)

mydb = mysql.connector.connect(host="localhost", user="root", password="", database="fund_advisor")
mycursor = mydb.cursor()


# Redirecting Home
@app.route('/')
def index():
  if 'user_id' in session:
    return render_template('index.html', log_in='log_in')
  else:
    return render_template('index.html', log_in = 'log_out')
@app.route('/auth/signup')
def signup():
  if 'user_id' in session:
    return redirect(url_for('dashboard'))
  else:
    return render_template('/auth/signup.html')


# after creating account
@app.route('/profile/success', methods=['POST','GET'])
def success():
  if request.method == "POST":
    username = request.form.get('username')
    email = request.form.get('email')
    username = request.form.get('username')
    password = request.form.get('password')
    mycursor.execute("insert into fund_advisor (name,email,password)values(%s,%s,%s)", (username,email,password))
    mydb.commit()
    return render_template('/profile/success.html', username=username)
  else:
    return redirect('/')


# login valid
@app.route('/login_validation', methods=['POST','GET'])
def profile():
  email = request.form.get('email')
  password = request.form.get('password')
  mycursor.execute("""SELECT * FROM `fund_advisor` WHERE `email` LIKE '{}' AND `password` LIKE '{}'""".format(email,password))

  users = mycursor.fetchall()
 
  if len(users)>0:
    session['user_id']=users[0][0]
    # global useme
    # global email_me

    session['name']=users[0][1]
    session['email']=users[0][2]

    # session['listing'] = users
    # useme = session['name']
    # email_me = session['email']

    return redirect(url_for('dashboard'))
  else:
    # flash redirect("/auth/login.html", error_show='true')
    session['listning']= 'exist'
    return redirect(url_for('login'))
    
  # return render_template('profile/dashboard.html')  





# questionnaire
@app.route('/profile/questionnaire')
def questionnaire():
  if 'user_id' in session:
    return render_template('/profile/questionnaire.html')
  else:
      return redirect('/')

# dashboard
@app.route('/profile/dashboard',  methods=['POST','GET'])
def dashboard():
  checking = session.get('checking')
  if 'user_id' in session:
    if request.method == "POST":

      # Calling values
      session['checking'] = "true"
      time_frame = int(request.form['time_frame'])
      david1_v = int(request.form['david1'])
      investment_purpose = int(request.form['pq3'])
      net_worth = int(request.form['pq4'])
      investment_patiance =  int(request.form['pq5'])
      investment_scenario = int(request.form['pq6'])
      financial_decision = int(request.form['pq7'])
      financial_situation = int(request.form['pq9'])
      future_plan = int(request.form['pq10'])
      financial_goal = int(request.form['pq11'])
      risk_free_rate = 0.0879
      confidence_interval = 0.70

      # Data
      islamic = pd.read_excel('Islamic.xlsx', header = [0,1], index_col=0)
      returns = pd.read_excel ('Islamic.xlsx' , sheet_name= 'Returns', header = [0,1] , index_col= 0)
      RF_Data = pd.read_excel ('Islamic.xlsx' ,  sheet_name='Risk Free ', index_col='Date')
      prices = pd.read_excel('marketcapislamic.xlsx')
      islamic_yearly_returns = pd.read_excel ('Islamic.xlsx' ,  sheet_name='yearly Rp',  header = [0,1], index_col=0)
      category = ['Shariah Compliant Aggressive Fixed Income & Income', 'Shariah Compliant Asset Allocation & Balanced', 'Shariah Compliant Capital Protected Fund', 'Shariah Compliant Commodities', 'Shariah Compliant Equity', 'Shariah Compliant Index Tracker', 'Shariah Compliant Money Market']
      conservative = ['Shariah Compliant Money Market','Shariah Compliant Capital Protected Fund']
      moderately_conservative = ['Shariah Compliant Aggressive Fixed Income & Income','Shariah Compliant Money Market']
      moderate = ['Shariah Compliant Aggressive Fixed Income & Income','Shariah Compliant Capital Protected Fund']
      moderately_aggressive = ['Shariah Compliant Asset Allocation & Balanced','Shariah Compliant Index Tracker']
      aggresssive = ['Shariah Compliant Equity','Shariah Compliant Commodities']




      if (time_frame >= 0 and time_frame < 3 ):
        time_frame = 1

      elif (time_frame >= 3 and time_frame < 5): 
        time_frame = 2
      
      elif (time_frame >= 5 and time_frame < 10 ): 
        time_frame = 3
      
      elif (time_frame >= 10 ):
        time_frame = 4


             
      if ( net_worth == 1 and financial_goal == 1 ):
        A = 5

      elif ( (net_worth == 1 and financial_goal == 2) or (net_worth == 2 and financial_goal == 1) ): 
        A = 4.5
      
      elif ( (net_worth  == 1 and financial_goal == 3)  or (net_worth == 3 and financial_goal == 1) or (net_worth == 2 and financial_goal == 2)): 
        A = 4
      
      elif ( ( net_worth  == 1 and financial_goal == 4 ) or (net_worth == 2 and financial_goal == 3) or (net_worth == 3 and financial_goal == 2) ):
        A = 3.5
      

      elif ( (net_worth  == 2 and financial_goal == 4)  or (net_worth == 3 and financial_goal == 3)): 
        A = 3

      elif ( (net_worth  == 3 and financial_goal == 4) ): 
        A = 2.5

      elif ( (net_worth  == 4 and financial_goal == 1) ): 
        A = 2

      elif ( (net_worth  == 4 and financial_goal == 2) ): 
        A = 1.5
      

      elif ( (net_worth  == 4 and financial_goal == 3) ): 
        A = 1
      

      elif ( (net_worth  == 4 and financial_goal == 4) ): 
        A = 0.5
      
      
      time_horizon = david1_v + investment_purpose + time_frame
      session['time_horizon'] = time_horizon
      risk_and_objective = investment_patiance + investment_scenario + financial_decision + financial_situation + future_plan + financial_goal
      session['risk_and_objective'] = risk_and_objective


      def modefied_sharpe_ratio_evaluation(islamic, RF_Data, category ):
        # yearlyreturns = islamic.resample('Y').ffill().pct_change()
        # islamic_yearly_returns = yearlyreturns.groupby(yearlyreturns.index.year).agg(np.mean)
        S_D = returns.groupby(returns.index.year).std() * np.sqrt(252)
        RF = RF_Data.groupby(RF_Data.index.year).mean()
        s = returns.groupby(returns.index.year).skew()
        k = returns.groupby(returns.index.year).apply(pd.DataFrame.kurt)
        z = st.norm.ppf(0.05)
        t=z+1/6.*(z**2-1)*s+1/24.*(z**3-3*z)*k-1/36.*(2*z**3-5*z)*s**2
        mvar = np.add(np.multiply(t,S_D),islamic_yearly_returns)
        msr = np.divide(np.subtract(islamic_yearly_returns,RF), mvar )
        msr.loc['avg'] = msr[:7].mean()
        modefiedsharpe = msr.drop(msr.index[0:6])
        ms_ratio = modefiedsharpe.sort_values(by='avg', ascending= False, axis=1)

        first_priority = []
        flat_list = []
        if (len(ms_ratio[category[0]].columns) and len(ms_ratio[category[1]].columns) != 1):
          for cate in category:
            first_priority.extend([ms_ratio[cate].T.index[0], ms_ratio[cate].T.index[1]])

        else:
          if len(ms_ratio[category[0]].columns) > len(ms_ratio[category[1]].columns):
            first_priority.extend([ms_ratio[category[0]].T.index[[0,1,2]], ms_ratio[category[1]].T.index[[0]]])
          else:
            first_priority.extend([ms_ratio[category[[0]]].T.index[0], ms_ratio[category[1]].T.index[[0,1,2]]])

          for sublist in first_priority:
            for item in sublist:
              flat_list.append(item)
          first_priority = flat_list

        return first_priority               


      if ( ((time_horizon >=3 and time_horizon <= 4.8) and (risk_and_objective >= 6 and risk_and_objective <= 16.8)) or ((time_horizon >=4.9 and time_horizon <= 6.6) and (risk_and_objective >= 6 and risk_and_objective <= 9.6))):
        use_cols = modefied_sharpe_ratio_evaluation(islamic, RF_Data, conservative)
          
      if ( ((time_horizon >=4.9 and time_horizon <= 6.6) and (risk_and_objective >= 9.7 and risk_and_objective <= 16.8)) or ((time_horizon >=6.7 and time_horizon <= 8.4) and (risk_and_objective >= 6 and risk_and_objective <= 13.2)) or ((time_horizon >= 8.5 and time_horizon <= 10.2) and (risk_and_objective >= 6 and risk_and_objective <= 9.6))):
        use_cols =  modefied_sharpe_ratio_evaluation(islamic, RF_Data, moderately_conservative)
          
      
      if (  ((time_horizon >=3 and time_horizon <= 4.8) and (risk_and_objective >= 16.9 and risk_and_objective <= 24)) or ((time_horizon >= 10.2 and time_horizon <= 12) and (risk_and_objective >= 6 and risk_and_objective <= 13.2))     or   ((time_horizon >=4.9 and time_horizon <= 6.6) and (risk_and_objective >= 16.9 and risk_and_objective <= 20.4)) or ((time_horizon >= 6.7 and time_horizon <= 8.4) and (risk_and_objective >= 13.3 and risk_and_objective <= 16.8)) or ((time_horizon >= 8.5 and time_horizon <= 10.2) and (risk_and_objective >= 9.7 and risk_and_objective <= 16.8)) or ((time_horizon >= 8.5 and time_horizon <= 10.2) and (risk_and_objective >= 9.7 and risk_and_objective <= 16.8))  ):
        use_cols =  modefied_sharpe_ratio_evaluation(islamic, RF_Data, moderate)
          
      if ( ((time_horizon >=4.9 and time_horizon <= 6.6) and (risk_and_objective >= 20.5 and risk_and_objective <= 24)) or ((time_horizon >= 6.7 and time_horizon <= 8.4) and (risk_and_objective >= 16.9 and risk_and_objective <= 20.4)) or ((time_horizon >= 8.5 and time_horizon <= 10.2) and (risk_and_objective >= 16.9 and risk_and_objective <= 20.4)) or ((time_horizon >= 10.2 and time_horizon <= 12) and (risk_and_objective >= 13.2 and risk_and_objective <= 16.8))   ):
        use_cols =  modefied_sharpe_ratio_evaluation(islamic, RF_Data, moderately_aggressive)
      
      
      if ( ((time_horizon >= 6.7 and time_horizon <= 8.4) and (risk_and_objective >= 20.5 and risk_and_objective <= 24))  or  ((time_horizon >= 8.5 and time_horizon <= 10.2) and (risk_and_objective >= 20.5 and risk_and_objective <= 24)) or ((time_horizon >= 10.2 and time_horizon <= 12) and (risk_and_objective >= 16.9 and risk_and_objective <= 24)) ):
        use_cols =  modefied_sharpe_ratio_evaluation(islamic, RF_Data, aggresssive)

      bench_markold = np.array(pd.read_excel('Benchmarkislamic.xlsx',sheet_name='benchmark')[use_cols])


      bench_mark = bench_markold.flatten()
      bench_mark
      investor_opinion = [0,0,0,0]
        
      funds = []
      for fund in returns:
        funds.append(fund[1])
      final_funds = pd.DataFrame([],index=returns.index, columns=funds)
      for category in returns:    
        for sub_fund in returns[category[0]]:
          final_funds[sub_fund] = returns[category[0]][sub_fund]
      final_funds
      no_of_funds = len(use_cols)

      def blacklitterman_model(df,bench_mark, prices, ticker, no_of_funds, confidence_interval, A,risk_free_rate, investor_opinion):

        df = final_funds[ticker]
        #Covariance
        cov = df.cov()*252

        #Tracking Factor
        tracking_factor = np.zeros((no_of_funds,no_of_funds))
        for i in range(no_of_funds):
          for j in range(no_of_funds):
            tracking_factor[i, j] = cov.iloc[i,j] / cov.iloc[i,i]
        tracking_factor = pd.DataFrame(tracking_factor, columns= ticker)

        #Black_litterman_model
        columns = ['Historical return','Historical Weight', 'Implied Return', 'Implied Weight','Analyst and Opinion', 'Opinion Adjusted Return', 'Final Weights']
        asset_allocation = pd.DataFrame(np.zeros((no_of_funds,len(columns))),columns=columns, index=ticker)

        #Historical_Return
        for tic in ticker:
          asset_allocation['Historical return'][tic] = df[tic].mean()*252

        #Historical_Weights
        matrix_1 = np.matmul(np.linalg.inv(cov),asset_allocation['Historical return'] - risk_free_rate)
        asset_allocation['Historical Weight'] = matrix_1/sum(matrix_1)

        #Implied_Return
        capital = prices[ticker].sum()/prices[ticker].sum().sum()
        asset_allocation['Implied Return'] = np.multiply(np.matmul(capital,cov),bench_mark - risk_free_rate) / np.matmul(capital, np.matmul(capital,cov)) + risk_free_rate

        #Implied Weights
        matrix_2 = np.matmul(np.linalg.inv(cov),asset_allocation['Implied Return'] - risk_free_rate)
        asset_allocation['Implied Weight'] = matrix_2/sum(matrix_2)

        #Analyst and Opinion
        asset_allocation['Analyst and Opinion'] = investor_opinion

        #Opinion Adjusted Return
        asset_allocation['Opinion Adjusted Return'] = asset_allocation['Implied Return'] + np.dot(asset_allocation['Analyst and Opinion'],tracking_factor)
        asset_allocation


          #Utility Maximization
        def objective(x):
          return (((1/2)*(A)*np.dot(x,np.dot(x,cov))) - np.dot(x,asset_allocation['Opinion Adjusted Return']))

        def constraint1(x):
          sum_eq = 1.0
          for i in range(no_of_funds):
            sum_eq = sum_eq - x[i]
          return sum_eq

        x0 = [0.25,0.25,0.25,0.25]

        bnds = tuple((random.uniform(0,0.3),random.uniform(0.51,0.99)) for tic in ticker)
        con1 = {'type': 'eq', 'fun': constraint1}
        solution = minimize(objective,x0, method='SLSQP', bounds=bnds,constraints=con1)
        asset_allocation['Final Weights'] = (solution.x)

        asset_allocation['Final Weights'] = asset_allocation['Final Weights']*100
        return asset_allocation 
    
      a = blacklitterman_model(final_funds,bench_mark, prices, use_cols, no_of_funds, confidence_interval, A,risk_free_rate,investor_opinion)

      session['fund1'] = a.index[0]
      session['fund2'] = a.index[1]        
      session['fund3'] = a.index[2]
      session['fund4'] = a.index[3]

      session['value1']= round(a['Final Weights'][0],2)
      session['value2'] = round(a['Final Weights'][1],2)    
      session['value3'] = round(a['Final Weights'][2],2)
      session['value4'] = round(a['Final Weights'][3],2)

      return render_template('/profile/dashboard.html', username=session['name'], email = session['email'], fill_survey = "true", fund1 = session['fund1'], fund2 = session['fund2'], fund3 = session['fund3'], fund4 = session['fund4'], value1 = session['value1'], value2 = session['value2'], value3 = session['value3'], value4 = session['value4'], time_horizon= session['time_horizon'], risk_and_objective = session['risk_and_objective'])
      # return render_template('/profile/dashboard.html', username=session['name'], email = session['email'], fill_survey = "true", time_frame = answers)
    elif checking == "true": 
      return render_template('/profile/dashboard.html', username=session['name'], email = session['email'], fill_survey = "true", fund1 = session['fund1'], fund2 = session['fund2'], fund3 = session['fund3'], fund4 = session['fund4'], value1 = session['value1'], value2 = session['value2'], value3 = session['value3'], value4 = session['value4'], time_horizon= session['time_horizon'], risk_and_objective = session['risk_and_objective'])
    else:
      return render_template('/profile/dashboard.html', username=session['name'], email = session['email'], fill_survey = "false") 
  else:
    return redirect('/')

    
# login valid or not checking here
@app.route('/auth/login')
def login():
  if 'user_id' in session:
    return redirect(url_for('dashboard'))
  else:
    listing = session.get('listning')
    if (listing == 'exist'):
      session.pop('listning')
      return render_template('/auth/login.html', show_me="true")
    elif (listing != 'exist'):
      return render_template('/auth/login.html')

# login out
@app.route('/logout')
def logout():
  session.pop('user_id')
  return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)