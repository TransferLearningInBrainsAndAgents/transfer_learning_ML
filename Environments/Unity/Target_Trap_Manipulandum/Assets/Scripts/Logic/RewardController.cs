
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RewardController : MonoBehaviour
{

    private float action_reward = 0;
    private float area_reward = 0;
    private bool port_reward = false;
        
    void Start()
    {
        EventManager.Instance.onRewardFromAction.AddListener(SaveNewRewardDueToAction);
        EventManager.Instance.onNeedingNewTotalReward.AddListener(GenerateNewTotalReward);
        EventManager.Instance.onRewardPortTouched.AddListener(SaveNewRewardDueToPortTouched);
    }

    private void GenerateNewTotalReward()
    {
        float totalReward = action_reward + area_reward + (port_reward ? RewardStructure.Instance.RewPortPokedCorrectly: 0f);
        //Debug.Log($"Total Reward = {totalReward}");
        EventManager.Instance.onRewardReady.Invoke(totalReward);
        action_reward = 0f;
        area_reward = 0f;
        port_reward = false;
           
    }

    private void SaveNewRewardDueToAction(float new_reward)
    {
        action_reward = new_reward;
    }

    // This is called from the Visual Scripting Graph in the VRGame Object,
    // by the Transition (Success State -> Running_Trial State), 
    // if there is a Reward Port touched (OnEnter) event.
    public void SaveNewRewardDueToPortTouched()
    {
        EventManager.Instance.onStopFeaturesSending.Invoke();
        port_reward = true;
        EventManager.Instance.onRedoFeaturesObservations.Invoke(2);
    }


    //This is called from the Visual Scripting Graph in the VRGame Object,
    // when the reward areas are crossed for the first time after n seconds
    public void SaveNewRewardDueToAreaTouched(string area_type)
    {
        EventManager.Instance.onStopFeaturesSending.Invoke();
        switch (area_type)
        {
            case string value when value.Contains("High"):
                area_reward = RewardStructure.Instance.AreaHighInterest;
                break;
            case string value when value.Contains("Med"):
                area_reward = RewardStructure.Instance.AreaMedInterest;
                break;
            case string value when value.Contains("Low"):
                area_reward = RewardStructure.Instance.AreaLowInterest;
                break;
        }
        EventManager.Instance.onRedoFeaturesObservations.Invoke((int)area_reward);
    }

    
}
