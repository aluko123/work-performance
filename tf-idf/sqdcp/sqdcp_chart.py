import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates



def generate_daily_trend_chart():
    """
    loads predicted SQDCP data and creates a grouped bar chart
    showing weekly trend for each SQDCP category.
    """
    try:
        df = pd.read_csv('predicted_sqdcp.csv')
    except FileNotFoundError:
        print("Error: 'predicted_sqdcp.csv' not found")
        return
    
    #--- define work days and weeks based on 5-day cycle --- 
    df['date_obj'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['date_obj'], inplace=True)
    
    if df.empty:
        print("No valid dates found in the data. Cannot generate trend chart.")
        return
    
    #df['adjusted_date'] = (df['datetime'] - pd.to_timedelta('6 hours 30 minutes')).dt.date

    # sort list of unique workdays
    unique_workdays = sorted(df['date_obj'].unique())
    # map unique workday to a day number (0, 1, 2, etc.)
    workday_map = {day: i for i, day in enumerate(unique_workdays)}
    df['day_num'] = df['date_obj'].map(workday_map)

    #assign "work week" based on 5-day cycles
    df['Work_Week'] = (df['day_num'] // 5) + 1


    # # define sub-dimension columns for each SQDCP category
    # sqdcp_sub_dimensions = {
    #     'Safety': ['Hazard Identification_Pred', 'Escalation Criteria Stated_Pred', 'Escalation Decision Made_Pred', 'Proactive Phrasing_Pred', 'PPE & Compliance Visibility_Pred', 'Safety Coaching Language_Pred', 'Emotional Awareness_Pred', 'Silence on Safety_Pred'],
    #     'Quality': ['Clear Defect Description_Pred', 'Root Cause Exploration_Pred', 'Trend Recognition_Pred', 'Voice of Operator Reflected_Pred', 'Turn-Taking Balance_Pred', 'Containment vs. Root Fix Split_Pred', 'Quality Tradeoff Framing_Pred'],
    #     'Delivery': ['Clear Deviation Stated_Pred', 'Projection Statement_Pred', 'Proactive Phrasing_Pred', 'Containment Action Identified_Pred', 'Accountability Assigned_Pred', 'Escalation Decision Made_Pred', 'Trend Recognition_Pred', 'Voice of Operator Reflected_Pred', 'Voice of Customer Reflected_Pred'],
    #     'Cost': ['Scrap/Waste Acknowledged_Pred', 'Rework Time Stated_Pred', 'Overtime Justified or Flagged_Pred', 'Downtime Cost Noted_Pred', 'Labor Allocation Awareness_Pred', 'Material Waste/Inventory Excess_Pred', 'Tool/Equipment Cost Impact_Pred', 'Cost vs. Risk Tradeoff Framing_Pred', 'Prioritization Based on Cost_Pred', 'Coaching Language on Efficiency_Pred', 'Voice of Customer Cost_Pred'],
    #     'People': ['Feedback Quality (Tier 1/2)_Pred', 'Participation Inclusivity_Pred', 'Emotional Intelligence in Language_Pred', 'Dialogue Invitation_Pred', 'Recognition of Contributions_Pred', 'New Hire Status Shared_Pred', 'Training Progress Shared_Pred', 'Training Matrix Awareness_Pred', 'Mentoring/Support Mentioned_Pred', 'Workload or Morale Reflected_Pred']
    # }

    sqdcp_cols = ['Saftey_Score_Pred', 'Quality_Score_Pred', 'Delivery_Score_Pred', 'Cost_Score_Pred', 'People_Score_Pred']
    
    #filter for columns existing in df
    existing_cols = [col for col in sqdcp_cols if col in df.columns]

    #calculate daily average for final scores
    daily_avg_scores_df = df.groupby('date_obj')[existing_cols].mean()


    # # -- calculate daily average for each category---
    # daily_scores_list = []
    # for date, group in df.groupby('date_obj'):
    #     daily_sum = {'date': date}
    #     for category, sub_cols in sqdcp_sub_dimensions.items():
    #         existing_cols = [col for col in sub_cols if col in group.columns]
    #         if existing_cols:
    #             daily_sum[category] = group[existing_cols].sum().sum()
    #     daily_scores_list.append(daily_sum)

    # daily_scores_df = pd.DataFrame(daily_scores_list).set_index('date')
    # daily_scores_df.set_index('date', inplace=True    
    #     for category, sub_cols in sqdcp_sub_dimensions):
    #     existing_cols = [col for col in sub_cols if col in df.columns]
    #     if existing_cols:
    #         df[f'{category}_Avg'] = df[existing_cols].mean(axis=1)

    # daily_avg_cols = [f'{cat}_Avg' for cat in sqdcp_sub_dimensions.keys() if f'{cat}_Avg' in df.columns]
    # daily_scores_df = df.groupby('adjusted_date')[daily_avg_cols].mean()
    # daily_scores_df.index = pd.to_datetime(daily_scores_df.index)

    if daily_avg_scores_df.empty:
        print("Could not process daily scores.")
        return


    #---Create the Line Chart ---
    plt.figure(figsize=(15, 8))
    for column in daily_avg_scores_df.columns:
        label_name = column.replace('_Pred', '').replace('Safety', 'Safety')
        plt.plot(daily_avg_scores_df.index, daily_avg_scores_df[column], marker='o', linestyle='-', label=label_name)


    # -- Add vertical lines for Week Seperators --
    # find the end of each 5-day week
    week_end_days = [day for i, day in enumerate(unique_workdays) if (i + 1) % 5 == 0 and i < len(unique_workdays) - 1]
    for week_end in week_end_days:
        plt.axvline(x=week_end + pd.to_timedelta('12 hours'), color='r', linestyle='--', alpha=0.7, label='Week End' if week_end == week_end_days[0] else "")


    # n_categories = len(weekly_scores_df.columns)
    # n_weeks = len(weekly_scores_df.index)
    # bar_width = 0.15
    # index = np.arange(n_weeks)
    
    # plt.figure(figsize=(14, 7))
    # for i, category in enumerate(weekly_scores_df.columns):
    #     plt.bar(index + i * bar_width, weekly_scores_df[category], bar_width, label=category)
    

    #Plot
    plt.title('Daily SQDCP Communication Trends', fontsize=16)
    plt.ylabel('Daily Average Score (1-5)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    #plt.xticks(index + bar_width * (n_categories-1)/2, weekly_scores_df.index)
    #plt.ylim(0, 5.5)
    plt.legend(title='SQDCP Category')
    plt.grid(True, which='both',linestyle='--', linewidth=0.5)
    #plt.tight_layout()

    #Format x-axis dates for readability
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()


    output_filename = 'sqdcp_daily_trends.png'
    plt.savefig(output_filename)
    plt.close()
    print(f"SQDCP daily trends chart saved as '{output_filename}'")

#run the function
if __name__ == "__main__":
    generate_daily_trend_chart()
