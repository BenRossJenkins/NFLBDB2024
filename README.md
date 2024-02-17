# NFLBDB2024

# Beyond the Tackle: Novel Metrics for Evaluating Defensive Performance in the NFL Using BiLSTM Networks

# Introduction

American football is a game of strategy and physical prowess. It presents unique challenges in understanding and quantifying defensive performance. Vince Lombardi once emphasized the essence of tackling: stopping the opponent by any means necessary. This principle underlies our innovative approach to analyzing tackling, a critical aspect of football defense.

We have crafted three novel metrics to provide a more nuanced understanding of defensive play. Our goal is to transcend traditional statistics by offering insightful, data-driven evaluations of tackling. We quantify the contribution of all defensive players towards the outcome of a play.

# Our Metrics

1. **Time Saved:** This metric evaluates a defender's efficiency in reducing the time to tackle the ball carrier. Utilizing a BiLSTM neural network, we analyze players' positions and movements to predict the expected time for a tackle and compare it with the actual time taken. **Time Saved** is a measurement of how efficient the tackle was.
2. **Optimal Path Deviation:** This quantifies the variance between a defender's actual path and the calculated optimal path to the ball carrier. It's a measure of path efficiency, highlighting players who adeptly navigate the field. 
3. **PURSUIT:** A dynamic measure of a defender's effectiveness in chasing and engaging with the ball carrier. It considers both the rate of closing distance and the angle of pursuit. **PURSUIT** and the **Optimal Path Deviation** are useful performance measures for all defensive players and are not limited to the player making the tackle.

Together these metrics offer a comprehensive view of defensive tackling performance and individual contributions. They shed light on the subtle, often overlooked aspects of defense, such as constraining the ball carrier's options or steering them into disadvantageous positions.

# **Time Saved**

We measure a defender's ability to reduce the time to tackle the ball carrier using a Bidirectional Long Short-Term Memory (BiLSTM) neural network. This network learns from relative player positions and movements, predicting the time a defender should take to reach the ball carrier. By processing both past and upcoming player movements, the BiLSTM network captures the dynamic interplay of the game, offering real-time assessments of defensive actions (**Figure 1**).

The network is trained on a feature set including relative positions, directions, and velocities of players, allowing it to understand and anticipate how defenders close in on ball carriers (**Table 1**). As it learns from each play's sequence, it refines its predictions of contact timing. We found a significant improvement in using relative coordinates instead of absolute and used a similar feature set by the 2020 NFL Big Data Bowl winners ([1](https://www.kaggle.com/competitions/nfl-big-data-bowl-2020/discussion/119400))
to predict rushing yards after handoff.

**Time Saved** is the difference between the expected time until contact and actual time taken by a defender to make the tackle, with positive values indicating faster-than-expected performance. This metric has proven effective in identifying skilled defenders and aligns with conventional tackling metrics and overall team defensive performance.

**Figure 2** illustrates the **Time Saved** calculation, featuring Ernest Jones tackling Josh Allen. At the specific time step shown, Ernest Jones executes a tackle precisely as predicted by the BiLSTM model, resulting in a **Time Saved** value of 0 seconds. To summarize player performance, we compute the mean **Time Saved** across the entire play. In this instance, Ernest Jones’ mean **Time Saved** of 0 seconds aligns with the BiLSTM model’s expectations. By aggregating **Time Saved** over multiple plays and the entire season, we gain a succinct overview of a player’s tackling efficiency. See **Figure 7** for the top 10 defensive players by position.

The BiLSTM model achieves remarkable accuracy, with an RMSE of 0.04 seconds, surpassing alternative methods such as gradient boosting and other neural networks. Its bidirectional analysis offers an improved perspective on player interactions and movements, which is pivotal for assessing and improving tackling efficiency in dynamic scenarios. For a deeper dive into model performance, refer to the **Appendix**.
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/BiLSTMmodel.jpg?raw=true)
**Figure 1: BiLSTM Network Architecture**

![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/TUCfinalfeatures.png?raw=true)
**Table 1: Features Used in BiLSTM to Predict Time Until Contact.**

![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/timesavedexample.png?raw=true)
**Figure 2: Time Until Contact Calculation**

Defensive players dynamically adjust their motion to pursue the ball carrier effectively. Using a BiLSTM neural network, we calculate the Optimal Angle of Pursuit (**AOP**) for each defender. **AOP** represents the most efficient path toward the ball carrier (see **Figure 3**). Updated every 0.1 seconds, **AOP** measures the deviation between a defender’s current direction and the projected ball carrier location, ranging from 0 to 180 degrees. Smaller **AOP** values indicate closer alignment with the ideal path, increasing the likelihood of successful tackles. This real-time angle guides defenders in optimizing their positioning.
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/angleofpursuitfinal.jpg?raw=true)
**Figure 3:  Angle of Pursuit Calculation**

# **Optimal Path Deviation** 

The Optimal Path Deviation (**OPD**) measures how closely a defender’s actual route aligns with the ideal trajectory derived from the Angle of Pursuit (**AOP**) toward the ball carrier. A smaller **OPD** indicates more efficient pursuit, increasing the likelihood of a successful tackle. This metric provides real-time insights into the accuracy of a defender’s movements and tactical positioning.

In **Figure 4**, we observe the Optimal Path Deviation (**OPD**) during a specific play. The yellow circle represents the ideal path, while the red circle traces the actual path taken by Ernest Jones. Although Jones successfully made the tackle, the deviation suggests that he could have reached the ball carrier more efficiently by adhering closely to the optimal path. This example highlights how **OPD** evaluates and guides a defender’s route efficiency in real-time scenarios.

# **PURSUIT** 

Top tacklers excel by minimizing the distance to the ball carrier in the shortest amount of time. Drawing from Quang Nguyen et al. (2023) ([2](https://arxiv.org/abs/2305.10262)), we introduce **PURSUIT** as a dynamic metric. It considers the rate of change in both distance and angle of pursuit. A defender’s **PURSUIT** increases with quicker, more direct approaches to the ball carrier and decreases with indirect or distant paths. This metric quantifies a defender’s effectiveness in trajectory and speed toward the ball carrier. It peaks at perfect pursuit and diminishes with less optimal angles. **PURSUIT** is computed frame-by-frame for each defender during a play according to:

$$\Huge \large \text{PURSUIT}_{ij}(t) =\begin{cases}
0 & \text{for } t = 1 \\
-\frac{f'_{dij}(t)}{d_{ij}(t)} \cdot \left(1 - \frac{\text{angle of pursuit}_{ij}(t)}{180}\right) & \text{for } t > 1
\end{cases}$$

Where:
- $\large \text{PURSUIT}_{ij}(t)$ represents the **PURSUIT** metric for defender $i$ at frame $t$.
- $\large f'_{dij}(t)$ is the rate of change (derivative) of the defender's distance to the ball carrier at frame $t$.
- $\large d_{ij}(t)$ is the absolute distance between the defender $i$ and the ball carrier at frame $t$.
- $\large \text{angle of pursuit}_{ij}(t)$ is the angle of pursuit between defender $i$ and the ball carrier at frame $t$.

**PURSUIT** is calculated based on the rate at which the distance between the defender and the ball carrier changes, the current distance, and the angle of pursuit. Here are some examples to illustrate its applications:

- Defenders who have a higher average **PURSUIT** throughout a game are better tacklers.
- A higher **PURSUIT** score in a play correlates with a higher chance of successfully tackling the ball carrier.
- Defenders who increase their **PURSUIT** during a play are more likely to reach and engage ball carriers who are attempting to evade them.
- **PURSUIT** identifies unsung defenders who make a positive contribution to plays but are not given credit in the stat sheet.

# Example Play

In a play during the Buffalo Bills versus LA Rams matchup (**Figure 4**), Ernest Jones demonstrates strategic awareness and agility. Initially positioned 11.3 yards away from Josh Allen, Jones adjusts his pursuit angle, gradually closing the gap. Jones executes a successful tackle and reaches the ball carrier at the time expected. His calculated approach highlights the dynamic decision-making process crucial for defensive effectiveness.
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/actualplay.gif?raw=true)
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/defenderanimation_optimalpath.gif?raw=true)
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/fourmetrics.png?raw=true)
**Figure 4: Play Animation Showing Metrics of Tackler**

Our approach offers a unique advantage: it allows us to assess the contributions of all defenders involved in a play, not just those officially credited with tackles or assists. This enables us to recognize defenders who might otherwise go unnoticed but play a crucial role in the play’s outcome. To illustrate this, we revisit the play, displaying all players on the field along with their **PURSUIT** values. The **PURSUIT** metric highlights the numerous defenders who positively impacted the play 
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/actualplay.gif?raw=true)
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/alldefendersplayanimationoptimalclose.gif?raw=true)
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/pursuitopdall.png?raw=True)
**Figure 5: Play Animation Showing PURSUIT and Optimal Path of All Defenders**

# Results

Our analysis reveals that the **PURSUIT** metric remains remarkably stable throughout the regular season, maintaining consistency from Week 1 to Week 9 (**Figure 6**). Moreover, a robust correlation (r = 0.79) and substantial explained variability (R-squared = 0.62) indicate that **PURSUIT** serves as a predictive indicator of future tackling performance. Players with high or low **PURSUIT** values in the early weeks are likely to maintain similar performance levels in subsequent weeks, establishing **PURSUIT** as a reliable metric for long-term tackling predictions.
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/PURSUITstabilityovertime.png?raw=true)
**Figure 6: Stability of PURSUIT from Weeks 1-5 to Weeks 6-9**

We present the top ten defenders, categorized by position, ranked according to our metrics: **Time Saved**, Optimal Path Deviation (**OPD**), and **PURSUIT** (**Figure 7**). We separate tackling performance into overall, run, and pass. By computing the mean value for each metric across all defenders and frames, we arrive at a single measure. These interconnected metrics collectively gauge tackling performance, and it’s noteworthy that the best tacklers consistently excel across all three. These defenders adeptly pursue opponents, minimizing the time required to reach the ball carrier. These top players were often recognized by Pro Bowl and All-Pro Honors. 
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/toplb.png?raw=true)
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/topdt.png?raw=true)
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/topdb.png?raw=true)
**Figure 7: Top 10 Defenders by Position**

We offer an insightful visualization of teams' tackling performance in both run and pass plays by leveraging the **PURSUIT** metric. **Figure 8** categorizes NFL teams into four distinct quadrants based on their run and pass tackling efficiency, measured through the **PURSUIT** score. This quadrant division allows us to observe and analyze patterns and disparities between teams' abilities to effectively tackle in run versus pass scenarios.

The horizontal axis represents the average **PURSUIT** score for pass plays, while the vertical axis corresponds to the same for run plays. Teams in the upper right quadrant excel in both run and pass tackling, indicating an effective defensive performance. Conversely, teams in the lower left quadrant exhibit room for improvement in both aspects. The other two quadrants represent teams that are stronger in either run tackling (upper left) or pass tackling (lower right).
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/pursuitteampassrun.png?raw=true)
**Figure 8: Team Pass and Run Tackling by PURSUIT**

We offer another visualization of teams' tackling performance in both run and pass plays with the **Time Saved** metric (**Figure 9**). A robust negative correlation exists between **Time Saved** on run and pass plays and the yards allowed per game for rushing and passing (r = -0.7 and -0.76, respectively). This suggests that teams with defenders consistently reaching ball carriers faster than anticipated tend to concede fewer yards, emphasizing the pivotal role of efficient tackling in curbing offensive production.
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/timesavedcorr.png?raw=true)
**Figure 9: Time Saved and Rushing and Passing Yards Allowed Per Game**

# Conclusion

While traditional metrics in the NFL evaluate defensive impact and tackling abilities, they often fall short by focusing solely on outcomes. Our novel metrics offer a more comprehensive view, capturing the efforts of defenders who influence play outcomes beyond direct tackles. By leveraging these metrics, we can uncover the hidden contributions of players who might otherwise remain unnoticed on the stat sheet. We firmly believe that these metrics will transform how we perceive and appreciate defensive prowess in the NFL.

However, it’s essential to recognize the limitations of our analysis. Firstly, these metrics emphasize individual actions and may overlook the intricate collaboration and tactical maneuvers that define successful defensive plays. While our current focus is on tackling, expanding these metrics to include other defensive actions—such as pass deflections, interception attempts, and coverage effectiveness—would provide a broader player impact perspective. Lastly, the effectiveness of these metrics can vary based on game situations and offensive/defensive strategies, necessitating ongoing refinement for wider applicability.

# Appendix

The BiLSTM model outperformed all other models tested, including linear regression, XGBoost, a CNN, and a vanilla LSTM. **Table 2** below shows the performance of various machine learning models examined and considered.

| Metrics            | Linear Regression | XGBoost  | CNN     | LSTM    | BiLSTM  |
|--------------------|-------------------|----------|---------|---------|---------|
| Train MSE          | 0.320             | 0.115    | 0.718   | 0.273   | **0.004**   |
| Train RMSE         | 0.566             | 0.339    | 0.847   | 0.522   | **0.064**   |
| Train MAE          | 0.254             | 0.142    | 0.623   | 0.231   | **0.031**   |
| Train R-Squared    | 0.839             | 0.942    | 0.717   | 0.865   | **0.998**   |
| Validation MSE     | 0.325             | 0.132    | 0.627   | 0.265   | **0.004**   |
| Validation RMSE    | 0.570             | 0.363    | 0.792   | 0.515   | **0.067**   |
| Validation MAE     | 0.255             | 0.150    | 0.546   | 0.235   | **0.033**   |
| Validation R-Squared | 0.839          | 0.934    | 0.758   | 0.869   | **0.998**   |

**Table 2: Comparison of Machine Learning Models Evaluated to Predict Time Until Contact. Bold values indicate best performance for that specific evaluation metric.**

We show the learning curve of the BiLSTM model per epoch. This helps to identify if the model is overfitting the data. The close proximity of the final RMSE values for both training and validation sets at the end of the observed epochs shows that the model has achieved a good balance between learning the training data patterns and maintaining performance on validation data. 
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/learrningcurve.png?raw=true)
**Figure 10: BiLSTM Learning Curve**

Unsurprisingly, the model's accuracy improves as the play progresses because there is more data regarding the relative and historical information of players on the field. The model improves over time reaching around 0.05 RMSE on average 1 second into a given play.
![Sample GIF](https://github.com/BenRossJenkins/NFLBDB2024/blob/main/RMSE.png?raw=true)
**Figure 11: BiLSTM Predicted vs Actual Time Until Contact RMSE**

**References**

1. https://www.kaggle.com/competitions/nfl-big-data-bowl-2020/discussion/119400
2. https://arxiv.org/abs/2305.10262

**Acknowledgements** We greatly appreciate the guidance and feedback from several experts in football analytics. These include; Seth Walder (Sports Analytics Writer at ESPN), Jordan Chipka (Head of Research & Development at Telemetry Sports), Patrick Ward (Research & Development at Seattle Seahawks), Amelia Probst (Data Scientist at PFF), Ekene Olekanma (Football R&D Coordinator at San Francisco 49ers), Zach Drapkin (Quantitative Analyst at Philadelphia Eagles), and Ishan Mehta (Football Data Scientist at Houston Texans).

**Code & Documentation**: [GitHub](https://github.com/BenRossJenkins/NFLBDB2024)
