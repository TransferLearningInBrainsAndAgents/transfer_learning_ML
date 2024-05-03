
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// This is a singleton that defines the events that are passed between the game's objects (not the ones defined and used within the Visual Scripting system if that is utilised)
/// </summary>
public class EventManager : MonoBehaviour
{
    public static EventManager Instance;

    public UnityEvent onStartClient;
    public UnityEvent onClientStarted;
    public UnityEvent onStopClient;
    public UnityEvent onClientStopped;
    public UnityEvent onNeedingNewPixelsObservation;
    public UnityEvent onNeedingNewTotalReward;
    public UnityEvent onLeftButtonPressed;
    public UnityEvent onRightButtonPressed;
    public UnityEvent onLeftButtonUnPressed;
    public UnityEvent onRightButtonUnPressed;
    public UnityEvent onReseting;
    public UnityEvent onResetDone;
    public UnityEvent onStopFeaturesSending;
    public UnityEvent onStepTaken;
    
    [System.Serializable]
    public class StringEvent : UnityEvent<string> { }
    public class IntEvent : UnityEvent<int> { }
    public class IntArrayEvent : UnityEvent<int[]> { }
    public class ByteArrayEvent : UnityEvent<byte[]> { }
    public class FloatEvent: UnityEvent<float>{ }
    public class FloatArrayEvent : UnityEvent<float[]> { }
    public class ListFOfByteArraysEvent: UnityEvent<List<byte[]>> { }
    public class ScreenResolutionEvent: UnityEvent<int, int> { }


    public StringEvent onUpdatedAction;
    public StringEvent onParametersChange;
    public StringEvent onBodyCollisionInArea;
    public FloatEvent onRewardFromAction;
    public FloatEvent onRewardFromRules;
    public FloatEvent onRewardReady;
    public FloatArrayEvent onAddRewardTimersToObservations;
    public IntArrayEvent onRewardRulesUpdate;
    public ByteArrayEvent onPixelsObservationReady;
    public FloatEvent onNewMoveSnapReceived;
    public FloatEvent onNewRotateSnapReceived;
    public ScreenResolutionEvent onNewScreenResolution;
    public ListFOfByteArraysEvent onFeaturesObservationReady;


    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;

            //ZMQ Client Events
            onStartClient = new();
            onClientStarted = new();
            onStopClient = new();
            onClientStopped = new();

            //Parameters Events
            onReseting = new();
            onResetDone = new();
            onParametersChange = new();
            onNewMoveSnapReceived = new();
            onNewRotateSnapReceived = new();
            onNewScreenResolution = new();

            //Action Events
            onLeftButtonPressed = new();
            onRightButtonPressed = new();
            onLeftButtonUnPressed = new();
            onRightButtonUnPressed = new();
            onUpdatedAction = new();
            onBodyCollisionInArea = new();

            //Observations Events
            onNeedingNewPixelsObservation = new();
            onPixelsObservationReady = new();
            onFeaturesObservationReady = new();
            onAddRewardTimersToObservations = new();
            onStopFeaturesSending = new();
;
            //Reward Events
            onStepTaken = new();
            onRewardRulesUpdate = new();
            onNeedingNewTotalReward = new();
            onRewardFromRules = new();
            onRewardFromAction = new();
            onRewardReady = new();
            
        }

        else
            Destroy(this);
    }
}
