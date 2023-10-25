using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Threading.Tasks;
using Tensornet;

namespace NeuralNetworkButGood
{

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    internal class GraphRenderer
    {
        private Point Size;

        private Bitmap baseImage;

        private Color GraphColor;
        private Color TextColor;

        const int LineThickness = 3;
        const int FontSize = 3;

        Pen pen;
        Brush fontBrush;
        private static Font font = new Font("Arial", FontSize);

        public GraphRenderer(Point imageSize, Color bgColor, Color graphColor, Color? textColor = null)
        {
            this.Size = imageSize;
            if(textColor == null)
                TextColor = Color.Black;
            else
                TextColor = textColor.Value;

            this.GraphColor = graphColor;
            pen = new Pen(GraphColor, LineThickness);
            fontBrush = new SolidBrush(TextColor);

            baseImage = new Bitmap(Size.X, Size.Y);
            ForEachBitmap(baseImage, (x,y) =>
            {
                baseImage.SetPixel(x,y, bgColor);
            });
        }

        public void GenerateImageFromFunction(int pointsToSample, string filePath, Func<float, float> graph, float domainMin = -2, float domainMax = 2)
        {
            float stepSize = (Math.Abs(domainMin) + Math.Abs(domainMax)) / pointsToSample;
            (float,float)[] Data = new (float,float)[pointsToSample];

            for(int i = 0; i < pointsToSample; i++)
            {
                float x = i * stepSize;
                Data[i] = (x,graph(x));
            }

            GenerateImage(Data, filePath);
        }

        public void GenerateImage((float,float)[] DataPairs, string filePath)
        {
            Array.Sort(DataPairs, (a,b) => a.Item1.CompareTo(b.Item1));
            
            Bitmap bitmap = (Bitmap)baseImage.Clone();

            float yMax = float.MinValue;
            float yMin = float.MaxValue;

            foreach(var tuple in DataPairs)
            {
                if(yMax < tuple.Item2)
                    yMax = tuple.Item2;
                if(yMin > tuple.Item2)
                    yMin = tuple.Item2;
            }

            using(Graphics g = Graphics.FromImage(bitmap))
            {
                PointF mins = new PointF(DataPairs[0].Item1,yMin);
                PointF max = new PointF(DataPairs[DataPairs.Length - 1].Item1,yMax);

                for(int i = 1; i < DataPairs.Length; i++)
                {
                    var prevTuple = DataPairs[i-1];
                    var tuple = DataPairs[i];
                    g.DrawLine(pen, 
                    TransformPoint(mins, max, new PointF(tuple.Item1, tuple.Item2)),
                    TransformPoint(mins, max, new PointF(prevTuple.Item1, prevTuple.Item2))
                    );
                }

                g.DrawString(
                    $"X: {Math.Round(mins.X, 2)}-{Math.Round(max.X, 2)}"
                    , font, fontBrush, new Point(0, bitmap.Height - font.Height));

                g.DrawString(
                    $"Y: {Math.Round(yMax, 2)}-{Math.Round(yMin, 2)}"
                    , font, fontBrush, new Point(0, 0));
            }

            bitmap.Save(filePath);
        }

        private PointF TransformPoint(PointF mins, PointF max, PointF value)
        {
            return new PointF(Transform(Size.X, mins.X, max.X, value.X), Transform(Size.Y, mins.Y, max.Y, value.Y));
        }


        private static float Transform(float size, float min, float max, float value)
        {
            float percent = (value - min) / (max - min);

            return size * percent;
        }

        private static void ForEachBitmap(Bitmap bmp, Action<int,int> onForEach)
        {
            for(int x = 0; x < bmp.Width; x++)
            {
                for(int y = 0; y < bmp.Height; y++)
                {
                    onForEach(x,y);
                }  
            }   
        }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    internal class NeuralNetworkVisualiser
    {
        private int sample;
        private float range;
        private int layer;
        Point Location {get;set;}
        private string outputPath;
        private GraphRenderer graphRenderer;

        public NeuralNetworkVisualiser(int sample, float range, Point Location, int layer, string outputPath)
        {
            this.layer = layer;
            this.sample = sample;
            this.range = range;
            this.Location = Location;
            this.outputPath = outputPath;
            this.graphRenderer = new GraphRenderer(new Point(256, 256), Color.Gray, Color.Blue, Color.Yellow);
        }

        public void SampleWeight(NeuralNetworkFast neuralNetwork, TrainingData data, int pointsToSample = 16)
        {
            ILayer activeLayer = neuralNetwork.Layers[layer];

            (Tensor<float>, Tensor<float>)[] TrainingDataRaw = data.GetDataInstance();

            graphRenderer.GenerateImageFromFunction(pointsToSample, GeneratePath(), 
            (newWeightValue) =>
            {
                float cost = 0;
                foreach(var tuple in TrainingDataRaw)
                {
                    cost += NeuralNetworkTrainer.MeanSquaredError(
                        neuralNetwork.Run(tuple.Item1), 
                        tuple.Item2
                        );
                }
                return cost / TrainingDataRaw.Length;
            });
        }

        private string GeneratePath()
        {
            return $"Layter-{layer}Row-{Location.X}Col-{Location.Y}.png";
        }
    }
}