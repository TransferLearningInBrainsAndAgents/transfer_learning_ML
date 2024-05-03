
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RewardController : MonoBehaviour
{

    private float action_reward = 0;
    private float rules_reward = 0;
    
        
    void Start()
    {
        EventManager.Instance.onRewardFromAction.AddListener(SaveNewRewardDueToAction);
        EventManager.Instance.onNeedingNewTotalReward.AddListener(GenerateNewTotalReward);
        EventManager.Instance.onRewardFromRules.AddListener(SaveNewRewardDueToRule);
    }

    private void GenerateNewTotalReward()
    {
        float totalReward = action_reward + rules_reward;
      
        EventManager.Instance.onRewardReady.Invoke(totalReward);
        action_reward = 0f;
        rules_reward = 0f;
           
    }

    private void SaveNewRewardDueToAction(float new_reward)
    {
        action_reward = new_reward;
    }


    public void SaveNewRewardDueToRule(float new_reward)
    {
        EventManager.Instance.onStopFeaturesSending.Invoke();

        rules_reward = new_reward;

        GenerateNewTotalReward();
    }

    
}
