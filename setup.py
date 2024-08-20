from setuptools import setup, find_packages

setup(
    name="key-frame-detector",
    version="2.0.0",  # Updated version number
    author="Joel Ibaceta",  # You may want to add your name here or create a new 'contributor' field
    author_email="mail@joelibaceta.com",  # You may want to update this
    license='MIT',
    description="An optimized python tool to extract key frame images from a video file",
    long_description="An optimized  tool to detect and extract key frame images from a video file, now with improved performance for large videos and additional features",
    url="https://github.com/skyler14/video-keyframe-extractor", 
    project_urls={
        'Source': 'https://github.com/skyler14/video-keyframe-extractor', 
        'Tracker': 'https://github.com/skyler14/video-keyframe-extractor/issues' 
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'opencv-python',
        'av',  # PyAV for video processing
        'scipy',  # For signal processing functions
    ],
    extras_require={
        'dev': [
            'matplotlib',  # For plotting in debug mode
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords='video keyframe detection opencv av pyav image-processing',
    entry_points={
        "console_scripts": [
            'key-frames-detector=KeyFrameDetector.cli:main'
        ]
    },
    python_requires='>=3.6',
)