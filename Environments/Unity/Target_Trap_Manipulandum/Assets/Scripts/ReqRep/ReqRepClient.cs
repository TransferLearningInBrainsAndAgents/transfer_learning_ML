
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;


public class ReqRepClient : MonoBehaviour
{
    public enum ClientStatus
    {
        Inactive,
        Activating,
        Active,
        Deactivating
    }

    [SerializeField] private string host;
    [SerializeField] private string init_port;
    [SerializeField] private string observation_data_port;

    private ReqRepListener _listener;
    private ClientStatus _clientStatus = ClientStatus.Inactive;

    private bool newPixelsObservationArrayReady = false;
    private bool newFeaturessObservationReady = false;
    private bool newRewardReady = false;

    private byte[] observationPixelsArray;
    private List<byte[]> observationFeaturesList;
    private float reward = 0f;

    private List<string> comProtocol;

    private void Start()
    {
        _listener = new ReqRepListener(host, init_port, observation_data_port, HandleInitMessage, HandleResponseMessage);
        _listener.InitialisationRequestMessage();

        EventManager.Instance.onStartClient.AddListener(OnStartClient);
        EventManager.Instance.onClientStarted.AddListener(() => _clientStatus = ClientStatus.Active);
        EventManager.Instance.onStopClient.AddListener(OnStopClient);
        EventManager.Instance.onClientStopped.AddListener(() => _clientStatus = ClientStatus.Inactive);

        EventManager.Instance.onPixelsObservationReady.AddListener(SaveNewPixelsObservation);
        EventManager.Instance.onFeaturesObservationReady.AddListener(SaveNewFeaturesObservation);
        EventManager.Instance.onRewardReady.AddListener(SaveNewReward);

        EventManager.Instance.onStopFeaturesSending.AddListener(StopFeaturesSending);

        comProtocol = GameObject.Find("Rat").GetComponent<CommunicationProtocol>().observationsComProtocol;

        observationFeaturesList = new List<byte[]>() {BitConverter.GetBytes(0.0f)};

        OnStartClient();
    }


    private void OnDestroy()
    {
        if (_clientStatus != ClientStatus.Inactive)
            OnStopClient();
    }


    // Functions that deal with starting and stopping the Request Reply Client
    private void OnStartClient()
    {
        _clientStatus = ClientStatus.Activating;
        _listener.Start();
    }

    private void OnStopClient()
    {
        _clientStatus = ClientStatus.Deactivating;
        _listener.Stop();
    }

    /// <summary>
    /// <c>HandleInitMessage</c> is called when the agent sents a response as an initial handshake between the agent's process and the environment's (this game) process.
    /// </summary>
    private void HandleInitMessage(string message)
    {
        Debug.Log(message);
    }

    /// <summary>
    /// <c>HandleResponseMessage</c> is called when the agent sents a request for an observation to the environment.
    /// </summary>
    /// <param name="message">the message from the agent describing what observation it needs. 
    /// The possible values should be defined in the CommunicationProtocol</param>
    private void HandleResponseMessage(string message, ResponseSocket repSocket)
    {
        //Debug.Log($"-- HandleResponseMessage Start with message: {message}");


        while (!newRewardReady || !newFeaturessObservationReady || !newPixelsObservationArrayReady) { }

        TimeSpan timeout = new(0, 0, 1);


        if (message == comProtocol[0]) // "Pixels"
        {
            SendReward(repSocket, timeout, true);
            SendPixels(repSocket, timeout, false);
        }

        if (message == comProtocol[1]) // "Features"
        {
            SendFeatures(repSocket, timeout);
            SendReward(repSocket, timeout, false);
        }
            

        if (message == comProtocol[2]) // "Everything"
        {
            SendFeatures(repSocket, timeout);
            SendReward(repSocket, timeout, true);
            SendPixels(repSocket, timeout, false);
        }

        newRewardReady = false;
        newFeaturessObservationReady = false;
        newPixelsObservationArrayReady = false;
        //Debug.Log("-- HandleResponseMessage End");
    }

    // Functions that deal with the different requests by the agent (like camera observations, feature observations and rewards)
    private void SendReward(ResponseSocket repSocket, TimeSpan timeout, bool more)
    {
        //Debug.Log($"---- Start Sending Reward = {reward}");
        while (!newRewardReady) { }
        string stringReward = reward.ToString();
        repSocket.TrySendFrame(timeout, stringReward, more);
        reward = 0f;
        newRewardReady = false;
        //Debug.Log("---- End Sending Reward");
    }

    private void SendFeatures(ResponseSocket repSocket, TimeSpan timeout)
    {
        //Debug.Log($"---- Start Sending Features, e.g. {observationFeaturesList[8]}");
        while (!newFeaturessObservationReady) { }
        foreach(byte[] feature in observationFeaturesList)
        {
            repSocket.TrySendFrame(timeout, feature, true);
        }

        newFeaturessObservationReady = false;
        //Debug.Log("---- End Sending Features");
    }

    private void SendPixels(ResponseSocket repSocket, TimeSpan timeout, bool more)
    {
        //Debug.Log($"---- Start Sending Pixels {observationPixelsArray.Length}");
        while (!newPixelsObservationArrayReady) {}

        repSocket.TrySendFrame(timeout, observationPixelsArray, more);

        newPixelsObservationArrayReady = false;

        //Debug.Log("---- End Sending Pixels");
    }

    // Functions that create the buffers for the different responses of the environment to the agent
    // (camera obesrvations, features observations and reward)
    // These functions are called from Events Invoked by the corresponding controller objects

    private void SaveNewPixelsObservation(byte[] array)
    {
        //Debug.Log("---- Start Saving Pixels");
        observationPixelsArray = array;
        newPixelsObservationArrayReady = true;
        //Debug.Log("---- End Saving Pixels");
    }

    private void SaveNewFeaturesObservation(List<byte[]> features)
    {
        //Debug.Log("---- Start Saving Features");
        observationFeaturesList = features;
        newFeaturessObservationReady = true;
        //Debug.Log("---- End Saving Features");
    }

    private void SaveNewReward(float new_reward)
    {
        //Debug.Log($"---- Start Saving Reward = {new_reward}");
        reward = new_reward;
        newRewardReady = true;
        //Debug.Log("---- End Saving Reward");
    }

    private void StopFeaturesSending()
    {
        newFeaturessObservationReady = false;
    }
}
        
