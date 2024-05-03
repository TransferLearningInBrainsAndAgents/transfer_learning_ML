using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Array2DEditor;
using System.Threading;
using MathNet;
using System;

public class GlobalRewardFunction : MonoBehaviour
{

    public double epsilon = 1e-5;
    public static int numberOfRules = 8;

    private int[] stepCounter = new int[2] { 0, 0 };

    //rhos
    private float[,] rewardTimers;

    //rs
    private int[,] rulesIndicators;

    //c: 
    public int[,,] historyCorrelations;

    //k: 
    public float[,,] historyCorrelationStepCounter;

    //Rs
    public float[] rulesReward;

    //Memory (C)
    public Array2DInt memory = new Array2DInt();

    //taus
    public float[] taus = new float[numberOfRules];

    


    // Start is called before the first frame update
    void Start()
    {
        EventManager.Instance.onRewardRulesUpdate.AddListener(UpdateRule);
        EventManager.Instance.onStepTaken.AddListener(UpdateReward);

        numberOfRules = GameObject.FindGameObjectWithTag("Rat").GetComponent<CommunicationProtocol>().numberOfRules;
        rewardTimers = new float[numberOfRules, 2];
        rulesIndicators = new int[numberOfRules, 2];
        historyCorrelations = new int[numberOfRules, numberOfRules, 2];
        historyCorrelationStepCounter = new float[numberOfRules, numberOfRules, 2];
        memory.GridSize.Set(numberOfRules, numberOfRules);
    }

    void UpdateRule(int[] rulesOn)
    {
       if(stepCounter[0] == stepCounter[1])
        {
            Thread.Sleep(1);
            EventManager.Instance.onStepTaken.Invoke();
            return;
        }

        stepCounter[0] = stepCounter[1];

        float[] RTemp = new float[numberOfRules];

        for (int i = 0; i < numberOfRules; i++)
        {
            //Initialise the temprary Rewards vector
            RTemp[i] = 0f;

            //Set up the rs
            rulesIndicators[i, 0] = rulesIndicators[i, 1];
            rulesIndicators[i, 1] = rulesOn[i];

            int sumOfMemoryOverJ = 0;
            float sumOfHistCorStepCounterOverJ = 0;
            for (int j = 0; j < numberOfRules; j++)
            {
                //c.append(copy.copy(c[t - 1]))
                historyCorrelations[i, j, 0] = historyCorrelations[i, j, 1];
                historyCorrelationStepCounter[i, j, 0] = historyCorrelationStepCounter[i, j, 1];

                /*
                 Set up the historyCorrelations (cs) 
                 for place_i in np.arange(len(r_placements)):
                    for place_j in np.arange(len(r_placements)):
                        c[t][place_i, place_j] = (np.max([rs[place_j, t], c[t-1][place_i, place_j]]) -
                                                 np.min([rs[place_i, t], c[t-1][place_i, place_j]])) * (C[place_i, place_j] > 0) * \
                                                 (1 - (1 if rho[place_i, t - 1] > epsilon else 0))

                        delta_c = np.abs(c[t-1][place_i, place_j] - c[t][place_i, place_j])
                        delta_k = (1 / np.max([C[place_i, place_j], 1]))
                        k[t][place_i, place_j] = np.max([c[t][place_i, place_j] * (delta_c + k[t-1][place_i, place_j] - delta_k), 0])
                 */
                int[] maxarray = new int[2] { rulesIndicators[j, 1], historyCorrelations[i, j, 0]};
                int otherRuleHasHappened = Mathf.Max(maxarray);

                int[] minarray = new int[2] { rulesIndicators[i, 1], historyCorrelations[i, j, 0]};
                int thisRuleHasHappened = Mathf.Max(minarray);

                int C = Convert.ToInt32(Convert.ToInt32(memory.GetCell(i, j)) > 0);

                int rhoIndicator = rewardTimers[i, 0] < epsilon ? 0 : 1;

                historyCorrelations[i, j, 1] = (otherRuleHasHappened - thisRuleHasHappened) * C * rhoIndicator;

                int deltaHistoryCor = Mathf.Abs(historyCorrelations[i, j, 0] - historyCorrelations[i, j, 1]);
                float deltaHistCorStepCounter = 1 / Mathf.Max(memory.GetCell(i, j), 1);

                historyCorrelationStepCounter[i, j, 1] = Mathf.Max(historyCorrelations[i, j, 1] * (deltaHistoryCor + historyCorrelationStepCounter[i, j, 0] - deltaHistCorStepCounter),
                                                                   0f);

                sumOfMemoryOverJ += historyCorrelations[i, j, 0];
                sumOfHistCorStepCounterOverJ += historyCorrelationStepCounter[i, j, 0];

            }
            //Set up the rewardTimers (rhos)
            //rho[place_i, t] = max([(1 - betas[place_i]) * (rho[place_i, t - 1] + rs[place_i, t] * np.sum(c[t-1][place_i, :])), 0])
            float decay = 1 - 1 / taus[i];
            float rhoMultiplier = rewardTimers[i, 0] + rulesIndicators[i, 1] * sumOfMemoryOverJ;
            rewardTimers[i, 1] = Mathf.Max(new float[2] { decay * rhoMultiplier, 0f });

            //Set up the temprary Rewards (R_temp)
            //if rs[place_i, t] <= rho[place_i, t] / (1 - betas[place_i]) and \
            //   rho[place_i, t - 1] < epsilon and np.sum(k[t - 1][place_i, :]) > 0:
            if ((rulesIndicators[i, 1] <= rewardTimers[i, 1] / decay) && rewardTimers[i, 0] < epsilon && sumOfHistCorStepCounterOverJ > 0)
            {
                RTemp[i] = rulesIndicators[i, 1] * rulesReward[i];
            }
        }

        float Reward = 0f;
        foreach(float rew in RTemp)
        {
            Reward += rew;
        }

        EventManager.Instance.onRewardFromRules.Invoke(Reward);
        float[] latestRewardTimers = new float[numberOfRules];
        for(int i=0; i<numberOfRules; i++)
        {
            latestRewardTimers[i] = rewardTimers[i, 1];
        }
        EventManager.Instance.onAddRewardTimersToObservations.Invoke(latestRewardTimers);
    }

    void UpdateReward()
    {
        stepCounter[1] += 1;

    }
}
