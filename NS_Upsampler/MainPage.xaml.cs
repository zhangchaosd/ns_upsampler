using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.Media.Capture.Frames;
using Windows.Media.Capture;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Windows.Graphics.Imaging;
using Windows.Media.MediaProperties;

using Windows.AI.MachineLearning;
using Windows.Storage;
using Windows.Media;
using System.Reflection;
using Windows.Storage.Streams;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;


// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace NS_Upsampler
{
    public class ModelInput
    {
        public TensorFloat Data { get; set; }
    }

    public class ModelOutput
    {
        public TensorFloat Output { get; set; }
    }

    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private MediaCapture _mediaCapture = null;
        private MediaFrameReader _reader = null;
        private FrameRenderer _previewRenderer = null;
        private FrameRenderer _outputRenderer = null;

        private int _frameCount = 0;

        private const int IMAGE_ROWS = 1080;
        private const int IMAGE_COLS = 1920;

        //private OpenCVHelper _helper;
        private LearningModel _learningModel = null;
        private LearningModelSession _session = null;
        private LearningModelBinding _binding = null;

        private InferenceSession _session2 = null;

        private DispatcherTimer _FPSTimer = null;
        public MainPage()
        {
            this.InitializeComponent();

            //_previewRenderer = new FrameRenderer(PreviewImage);
            _outputRenderer = new FrameRenderer(OutputImage);

            //_helper = new OpenCVHelper();

            _FPSTimer = new DispatcherTimer()
            {
                Interval = TimeSpan.FromSeconds(1)
            };
            _FPSTimer.Tick += UpdateFPS;
        }

        private void UpdateFPS(object sender, object e)
        {
            var frameCount = Interlocked.Exchange(ref _frameCount, 0);
            FPSMonitor.Text = "FPS: " + frameCount;
        }

        private async Task InitializeMediaCaptureAsync(MediaFrameSourceGroup sourceGroup)
        {
            if (_mediaCapture != null)
            {
                return;
            }

            _mediaCapture = new MediaCapture();
            var settings = new MediaCaptureInitializationSettings()
            {
                SourceGroup = sourceGroup,
                SharingMode = MediaCaptureSharingMode.ExclusiveControl,
                StreamingCaptureMode = StreamingCaptureMode.Video,
                MemoryPreference = MediaCaptureMemoryPreference.Cpu
            };
            await _mediaCapture.InitializeAsync(settings);
        }

        private async Task CleanupMediaCaptureAsync()
        {
            if (_mediaCapture != null)
            {
                await _reader.StopAsync();
                _reader.FrameArrived -= ColorFrameReader_FrameArrivedAsync;
                _reader.Dispose();
                _mediaCapture = null;
            }
        }

        protected override async void OnNavigatedTo(NavigationEventArgs e)
        {
            //rootPage = MainPage.Current;

            // setting up the combobox, and default operation
            //OperationComboBox.ItemsSource = Enum.GetValues(typeof(OperationType));
            //OperationComboBox.SelectedIndex = 0;
            //currentOperation = OperationType.Blur;

            // Find the sources 
            var allGroups = await MediaFrameSourceGroup.FindAllAsync();
            var sourceGroups = allGroups.Select(g => new
            {
                Group = g,
                SourceInfo = g.SourceInfos.FirstOrDefault(i => i.SourceKind == MediaFrameSourceKind.Color)
            }).Where(g => g.SourceInfo != null).ToList();

            if (sourceGroups.Count == 0)
            {
                // No camera sources found
                return;
            }
            var selectedSource = sourceGroups.FirstOrDefault();

            // Initialize MediaCapture
            try
            {
                await InitializeMediaCaptureAsync(selectedSource.Group);
            }
            catch (Exception exception)
            {
                Debug.WriteLine("MediaCapture initialization error: " + exception.Message);
                await CleanupMediaCaptureAsync();
                return;
            }


            _learningModel = await LoadModelAsync();
            LearningModelDevice device = new LearningModelDevice(LearningModelDeviceKind.DirectXHighPerformance);
            _session = new LearningModelSession(_learningModel, device);
            _binding = new LearningModelBinding(_session);
            //_session2 = new InferenceSession("Assets/SRNet5.onnx");
            _session2 = new InferenceSession("Assets/SRNet5.onnx", SessionOptions.MakeSessionOptionWithCudaProvider(0));

            // Create the frame reader
            MediaFrameSource frameSource = _mediaCapture.FrameSources[selectedSource.SourceInfo.Id];
            BitmapSize size = new BitmapSize() // Choose a lower resolution to make the image processing more performant
            {
                Height = IMAGE_ROWS,
                Width = IMAGE_COLS
            };
            _reader = await _mediaCapture.CreateFrameReaderAsync(frameSource, MediaEncodingSubtypes.Bgra8, size);
            _reader.FrameArrived += ColorFrameReader_FrameArrivedAsync;
            await _reader.StartAsync();

            _FPSTimer.Start();
        }

        protected override async void OnNavigatedFrom(NavigationEventArgs args)
        {
            _FPSTimer.Stop();
            await CleanupMediaCaptureAsync();
        }

        byte[] ImageToByte(SoftwareBitmap bitmap)
        {
            byte[] bytes;
            using (var stream = new InMemoryRandomAccessStream())
            {
                var encoder = BitmapEncoder.CreateAsync(BitmapEncoder.BmpEncoderId, stream).AsTask().Result;
                encoder.SetSoftwareBitmap(bitmap);
                encoder.FlushAsync().AsTask().Wait();
                bytes = new byte[stream.Size];
                stream.AsStream().Read(bytes, 0, bytes.Length);
            }
            return bytes;
        }

        float[] ConvertToModelInput(byte[] imageBytes)
        {
            int width = 1920;
            int height = 1080;
            float[] floatValues = new float[3 * height * width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int pos = y * width + x;
                    int bgraPos = pos * 4;  // 因为是BGRA格式

                    floatValues[pos] = imageBytes[bgraPos + 2] / 255.0f;             // R
                    floatValues[pos + height * width] = imageBytes[bgraPos + 1] / 255.0f;  // G
                    floatValues[pos + 2 * height * width] = imageBytes[bgraPos] / 255.0f;   // B
                }
            }

            return floatValues;
        }

        // 以下是可能需要的辅助函数
        // 这只是一个示意性的函数，您可能需要根据实际需求进行适当的调整
        byte[] ConvertToByteFormat(float[] modelOutput)
        {
            byte[] result = new byte[modelOutput.Length]; // 乘以 4 为了转换为 BGRA8
            for (int i = 0; i < modelOutput.Length; i++)
            {
                byte value = (byte)(modelOutput[i] * 255); // 假设模型的输出是 [0, 1] 范围的浮点数
                result[i] = value;     // B
                //result[i * 4 + 1] = value; // G
                //result[i * 4 + 2] = value; // R
                //result[i * 4 + 3] = 255;   // A
            }
            return result;
        }

        private async void ColorFrameReader_FrameArrivedAsync(MediaFrameReader sender, MediaFrameArrivedEventArgs args)
        {
            var frame = sender.TryAcquireLatestFrame();
            if (frame != null)
            {
                SoftwareBitmap originalBitmap = null;
                var inputBitmap = frame.VideoMediaFrame?.SoftwareBitmap;
                if (inputBitmap != null)
                {
                    // The XAML Image control can only display images in BRGA8 format with premultiplied or no alpha
                    // The frame reader as configured in this sample gives BGRA8 with straight alpha, so need to convert it
                    originalBitmap = SoftwareBitmap.Convert(inputBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);

                    var byteData = ImageToByte(originalBitmap);
                    //float[] modelInputData = ConvertToModelInput(byteData);
                    var modelInput = TensorUInt8Bit.CreateFromArray(new long[] { 1080, 1920, 4 }, byteData.Take(1920*1080*4).ToArray());


                    //_binding.Bind("modelInput", modelInput);
                    //var results = _session.Evaluate(_binding, "modelOutput");

                    // 1. 从模型输出获取数据
                    //var outputTensor = results.Outputs["modelOutput"] as TensorUInt8Bit;
                    //var outputBytes = outputTensor.GetAsVectorView().ToArray();
                    //var outputBytes = byteData;

                    // 2. 将此数据转换为适合 SoftwareBitmap 的字节格式（通常为 BGRA8）
                    // 此处，我们假设 ConvertToByteFormat 函数将模型的 float 输出转换为 BGRA8 格式的字节数组
                    //byte[] outputBytes = ConvertToByteFormat(outputArray);

                    //sesion2
                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("modelInput", new DenseTensor<byte>(byteData.Take(1920*1080*4).ToArray(), new int[] { 1080, 1920, 4 }))
                    };

                    // session2

                    // 推理
                    var results = _session2.Run(inputs);
                    //var output = results["32"].AsEnumerable<float>();
                    var resultsArray = results.ToArray();
                    var outputBytes = resultsArray[0].AsTensor<byte>().ToArray<byte>();

                    // 处理输出...
                    //foreach (var item in results)
                    //{
                    //    Console.WriteLine(item);
                    //}

                    // 3. 使用这些字节数据更新 outputBitmap
                    SoftwareBitmap outputBitmap = new SoftwareBitmap(BitmapPixelFormat.Bgra8, 3840, 2160, BitmapAlphaMode.Premultiplied);
                    using (BitmapBuffer buffer = outputBitmap.LockBuffer(BitmapBufferAccessMode.Write))
                    {
                        using (var reference = buffer.CreateReference())
                        {
                            if (reference is IMemoryBufferByteAccess byteAccess)
                            {
                                
                                unsafe
                                {
                                    byte* data;
                                    uint capacity;
                                    byteAccess.GetBuffer(out data, out capacity);
                                    System.Runtime.InteropServices.Marshal.Copy(outputBytes, 0, (IntPtr)data, outputBytes.Length);
                                    //BitmapPlaneDescription bufferLayout = buffer.GetPlaneDescription(0);
                                    //for (int i = 0; i < bufferLayout.Height; i++)
                                    //{
                                    //    for (int j = 0; j < bufferLayout.Width; j++)
                                    //    {
                                    //        data[bufferLayout.StartIndex + bufferLayout.Stride * i + 4 * j + 0] = outputBytes[(2160 - i - 1) * 3840 + j];
                                    //        data[bufferLayout.StartIndex + bufferLayout.Stride * i + 4 * j + 1] = outputBytes[3840*2160 + (2160 - i - 1) * 3840 + j];
                                    //        data[bufferLayout.StartIndex + bufferLayout.Stride * i + 4 * j + 2] = outputBytes[3840 * 2160 * 2 + (2160 - i - 1) * 3840 + j];
                                    //        data[bufferLayout.StartIndex + bufferLayout.Stride * i + 4 * j + 3] = (byte)255;
                                    //    }
                                    //}
                                }
                            }
                        }
                    }




                    // Display both the original bitmap and the processed bitmap.
                    //_previewRenderer.RenderFrame(originalBitmap);
                    _outputRenderer.RenderFrame(outputBitmap);
                }

                Interlocked.Increment(ref _frameCount);
            }
        }
        private async Task<LearningModel> LoadModelAsync()
        {
            var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri("ms-appx:///Assets/SRNet5.onnx"));
            return await LearningModel.LoadFromStorageFileAsync(modelFile);
        }
    }
}
