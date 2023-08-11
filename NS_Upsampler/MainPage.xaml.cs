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

        private DispatcherTimer _FPSTimer = null;
        public MainPage()
        {
            this.InitializeComponent();

            _previewRenderer = new FrameRenderer(PreviewImage);
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
            _session = new LearningModelSession(_learningModel);
            _binding = new LearningModelBinding(_session);

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

                    //var modelInput = new ModelInput();
                    var byteData = ImageToByte(originalBitmap);
                    float[] modelInputData = ConvertToModelInput(byteData);
                    var modelInput = TensorFloat.CreateFromArray(new long[] { 1, 3, 1080, 1920 }, modelInputData);


                    SoftwareBitmap outputBitmap = new SoftwareBitmap(BitmapPixelFormat.Bgra8, originalBitmap.PixelWidth, originalBitmap.PixelHeight, BitmapAlphaMode.Premultiplied);


                    //VideoFrame inputVideoFrame = VideoFrame.CreateWithSoftwareBitmap(originalBitmap);

                    //var modelInput = new ModelInput(); // This should be based on your model's expected input type
                    //modelInput.data = inputVideoFrame;
                    


                    float[] inputData = new float[1 * 3 * 1080 * 1920];
                    // ... 填充 inputData ...

                    var inputTensor = TensorFloat.CreateFromArray(new long[] { 1, 3, 1080, 1920 }, inputData);

                    _binding.Bind("modelInput", modelInput);
                    var results = _session.Evaluate(_binding, "modelOutput");
                    //_binding.Bind("modelInput", inputTensor);
                    //var results = _session.Evaluate(_binding, "modelOutput");

                    //binding.Bind("onnx::Concat_0", modelInput);
                    //var results = await session.EvaluateAsync(binding, "SessionRun");

                    // Operate on the image in the manner chosen by the user.
                    //if (currentOperation == OperationType.Blur)
                    //{
                    //    _helper.Blur(originalBitmap, outputBitmap);
                    //}
                    //else if (currentOperation == OperationType.HoughLines)
                    //{
                    //    _helper.HoughLines(originalBitmap, outputBitmap);
                    //}
                    //else if (currentOperation == OperationType.Contours)
                    //{
                    //    _helper.Contours(originalBitmap, outputBitmap);
                    //}
                    //else if (currentOperation == OperationType.Histogram)
                    //{
                    //    _helper.Histogram(originalBitmap, outputBitmap);
                    //}
                    //else if (currentOperation == OperationType.MotionDetector)
                    //{
                    //    _helper.MotionDetector(originalBitmap, outputBitmap);
                    //}

                    // Display both the original bitmap and the processed bitmap.
                    _previewRenderer.RenderFrame(originalBitmap);
                    _outputRenderer.RenderFrame(outputBitmap);
                }

                Interlocked.Increment(ref _frameCount);
            }
        }
        private async Task<LearningModel> LoadModelAsync()
        {
            var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri("ms-appx:///Assets/sd.onnx"));
            return await LearningModel.LoadFromStorageFileAsync(modelFile);
        }
    }
}
