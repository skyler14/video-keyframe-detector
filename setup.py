from setuptools import setup, find_packages

setup(
    name="key-frame-detector",
    version="2.0.0",  # Updated version number
    author="Joel Ibaceta",  # You may want to add your name here or create a new 'contributor' field
    author_email="mail@joelibaceta.com",  # You may want to update this
    license='MIT',
    description="An optimized python tool to extract key frame images from a video file",
    long_description="An improved tool to detect and extract key frame images from a video file, now with enhanced performance for large videos and additional features",
    url="https://github.com/yourusername/video-keyframe-extractor",  # Update with your fork's URL
    project_urls={
        'Source': 'https://github.com/skyler14/video-keyframe-extractor',  # Update with your fork's URL
        'Tracker': 'https://github.com/skyler14/video-keyframe-extractor/issues'  # Update with your fork's issues URL
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'opencv-python',
        'numpy',
        'peakutils',
        'matplotlib',
        'ffmpeg-python',
        'ffprobe-python'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='video key-frame terminal ffmpeg extractor optimization',
    entry_points={
        "console_scripts": [
            'key-frames-detector=cli:main'
        ]
    }
)