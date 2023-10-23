using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using MathNet;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;

namespace NeuralNetworkButGood
{
    internal class NeuralNetwork
    {
        public float Cost = 0;

        public Layer[] Layers;
        private bool hasInitalised = false;
        public NeuralNetwork()
        {
            Layers = new Layer[0];
        }

        public NeuralNetwork(NeuralNetwork neuralNetwork)
        {
            Layers = new Layer[neuralNetwork.Layers.Length];

            for(int i = 0; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(neuralNetwork.Layers[i].GetLength());
            }
        }

        public static NeuralNetwork Copy(NeuralNetwork network)
        {
            NeuralNetwork copy = new NeuralNetwork(network);

            for (int i = 1; i < copy.Layers.Length; i++)
            {
                network.Layers[i].GetInfoCopy(out var w, out var b);
                copy.Layers[i].Populate(copy.Layers[i-1], w, b);
            }

            copy.InitaliseManual();
            return copy;
        }

        public void AddLayer(int Length)
        {
            Layer[] newLayers = new Layer[Layers.Length + 1];
            for(int i = 0; i < newLayers.Length -1; i++)
            {
                newLayers[i] = Layers[i];
            }
            newLayers[newLayers.Length - 1] = new Layer(Length);
            Layers = newLayers;
        }
        public void AddLayer(Layer layer)
        {
            Layer[] newLayers = new Layer[Layers.Length + 1];
            for (int i = 0; i < newLayers.Length - 1; i++)
            {
                newLayers[i] = Layers[i];
            }
            newLayers[newLayers.Length - 1] = layer;
            Layers = newLayers;
        }

        public void InitaliseRandom()
        {
            if(hasInitalised)
            {
                throw new Exception("Network has already been intialised");
            }
            hasInitalised = true;
            Layers[0].Populate(null, true);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].Populate(Layers[i-1], true);
            }
        }
        public void InitaliseManual()
        {
            if (hasInitalised)
            {
                throw new Exception("Network has already been intialised");
            }
            hasInitalised = true;
            Layers[0].Populate(null, false);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].Populate(Layers[i - 1], false);
            }
        }

        public float[] Run(float[] input)
        {
            if (!hasInitalised)
            {
                throw new Exception("Network has not been intialised");
            }

            Vector<float> Carriage = Vector<float>.Build.DenseOfArray(input);
            for(int i = 0; i < Layers.Length; i++)
            {
                Carriage = Layers[i].FeedForward(Carriage);
            }
            return Carriage.AsArray();
        }
        public Vector<float> Run(Vector<float> Carriage)
        {
            if (!hasInitalised)
            {
                InitaliseManual();
            }

            for (int i = 0; i < Layers.Length; i++)
            {
                Carriage = Layers[i].FeedForward(Carriage);
            }
            return Carriage;
        }
    }

    internal class Layer
    {
        private int LayerLength;
        private int LayerWidth;

        public Matrix<float> Weights;
        public Vector<float> Bias;

        private bool IsStartingLayer = false;
        private Func<float, float> activationFunction;
        public Layer(int length)
        {
            LayerLength = length;
        }

        public void Populate(Layer? PreviousLayer, bool UseRandomPop, Func<float, float>? activation = null)
        {
            if(PreviousLayer == null)
            {
                IsStartingLayer = true;
                return;
            }

            int prevLayerSize = PreviousLayer.LayerLength;

            LayerWidth = PreviousLayer.LayerLength;

            if(UseRandomPop)
            {
                Weights = Matrix<float>.Build.Random(LayerLength, LayerWidth);
                Bias = Vector<float>.Build.Random(LayerLength);
            }

            if (activation == null)
                this.activationFunction = RELU;
            else
                activationFunction = activation;
        }

        public void Populate(Layer? PreviousLayer, Matrix<float> copyWeights, Vector<float> copyBiases, Func<float, float>? activation = null)
        {
            if (PreviousLayer == null)
            {
                IsStartingLayer = true;
                return;
            }

            int prevLayerSize = PreviousLayer.LayerLength;

            LayerWidth = PreviousLayer.LayerLength;

            Weights = copyWeights;
            Bias = copyBiases;

            if (activation == null)
                this.activationFunction = ATAN;
            else
                activationFunction = activation;
        }

        public Vector<float> FeedForward(Vector<float> PrevLayerInput)
        {
            if (IsStartingLayer)
            {
                return PrevLayerInput;
            }

            Vector<float> weightedSum = Weights * PrevLayerInput;
            weightedSum += Bias;

            Vector<float> LayerOutput = Vector<float>.Build.SameAs(weightedSum);
            weightedSum.Map(activationFunction, LayerOutput);

            return LayerOutput;
        }

        public static float RELU(float input)
        {
            return Math.Max(input, 0);
        }
        public static float ATAN(float input)
        {
            return MathF.Atan(input);
        }
        public int GetLength()
        {
            return LayerLength;
        }

        public void GetInfoCopy(out Matrix<float> weights, out Vector<float> biases)
        {
            weights = Matrix<float>.Build.SameAs(Weights);
            Weights.CopyTo(weights);

            biases = Vector<float>.Build.SameAs(Bias);
            Bias.CopyTo(biases);
        }
    }
}
