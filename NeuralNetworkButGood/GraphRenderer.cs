using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Threading.Tasks;
using Tensornet;
using System.Runtime.CompilerServices;

namespace NeuralNetworkButGood
{

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    internal class GraphRenderer
    {
        private Point Size;

        private Bitmap baseImage;

        private Color GraphColor;
        private Color TextColor;

        const int LineThickness = 2;
        const int FontSize = 8;

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
            NetworkUtils.ForEachBitmap(baseImage, (x,y) =>
            {
                baseImage.SetPixel(x,y, bgColor);
            });
        }

        public void GenerateImageFromFunction(int pointsToSample, string filePath, Func<float, float> graph, float domainMin = -2, float domainMax = 2)
        {
            float stepSize = (Math.Abs(domainMin) + Math.Abs(domainMax)) / (pointsToSample - 1);
            (float,float)[] Data = new (float,float)[pointsToSample];

            for(int i = 0; i < pointsToSample; i++)
            {
                float x = i * stepSize + domainMin;
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

                if(yMin != yMax)
                    for(int i = 1; i < DataPairs.Length; i++)
                {
                    var prevTuple = DataPairs[i-1];
                    var tuple = DataPairs[i];
                    var p1 = TransformPoint(mins, max, new PointF(tuple.Item1, tuple.Item2));
                    var p2 = TransformPoint(mins, max, new PointF(prevTuple.Item1, prevTuple.Item2));

                    if(float.IsNaN(p1.X) || float.IsNaN(p2.X) || float.IsNaN(p1.Y) || float.IsNaN(p2.Y))
                    {
                        var p1a = TransformPoint(mins, max, new PointF(tuple.Item1, tuple.Item2));
                        var p2a = TransformPoint(mins, max, new PointF(prevTuple.Item1, prevTuple.Item2));
                    }

                    g.DrawLine(pen, p1, p2);
                }

                g.DrawString(
                    $"X: [{mins.X}, {max.X}]"
                    , font, fontBrush, new Point(0, bitmap.Height - font.Height));

                g.DrawString(
                    $"Y: [{yMin}, {yMax}]"
                    , font, fontBrush, new Point(0, 0));
            }

            bitmap.Save(filePath);
        }

        private PointF TransformPoint(PointF mins, PointF max, PointF value)
        {
            return new PointF(Transform(Size.X, mins.X, max.X, value.X), Size.Y - Transform(Size.Y, mins.Y, max.Y, value.Y));
        }


        private static float Transform(float size, float min, float max, float value)
        {
            float percent = (value - min) / (max - min);

            return size * percent;
        }
    }
}