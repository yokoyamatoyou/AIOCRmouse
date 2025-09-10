"""Enhanced preprocessing functionality tests."""

import pytest
import numpy as np
import cv2
from unittest.mock import patch

from src.core import preprocess


class TestImageQualityAssessment:
    """Test image quality assessment functionality."""
    
    def test_excellent_quality_image(self):
        """Test assessment of high quality image."""
        # Create a high quality test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Add some structure to make it more realistic
        image[20:80, 20:80] = 255
        image[30:70, 30:70] = 0
        
        quality, metrics = preprocess.assess_image_quality(image)
        
        assert isinstance(quality, preprocess.ImageQuality)
        assert "contrast" in metrics
        assert "sharpness" in metrics
        assert "noise_level" in metrics
        assert "brightness" in metrics
        assert "edge_density" in metrics
    
    def test_poor_quality_image(self):
        """Test assessment of low quality image."""
        # Create a low quality test image (uniform noise)
        image = np.random.randint(120, 130, (100, 100, 3), dtype=np.uint8)
        
        quality, metrics = preprocess.assess_image_quality(image)
        
        assert isinstance(quality, preprocess.ImageQuality)
        # Should be poor quality due to low contrast and structure
        assert quality in [preprocess.ImageQuality.POOR, preprocess.ImageQuality.FAIR]


class TestPreprocessingFunctions:
    """Test individual preprocessing functions."""
    
    def test_enhance_contrast_clahe(self):
        """Test CLAHE contrast enhancement."""
        image = np.random.randint(50, 100, (100, 100), dtype=np.uint8)
        enhanced = preprocess.enhance_contrast(image, "clahe")
        
        assert enhanced.shape == image.shape
        assert enhanced.dtype == np.uint8
    
    def test_enhance_contrast_histogram(self):
        """Test histogram equalization."""
        image = np.random.randint(50, 100, (100, 100), dtype=np.uint8)
        enhanced = preprocess.enhance_contrast(image, "histogram")
        
        assert enhanced.shape == image.shape
        assert enhanced.dtype == np.uint8
    
    def test_denoise_bilateral(self):
        """Test bilateral filtering denoising."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        denoised = preprocess.denoise_image(image, "bilateral")
        
        assert denoised.shape == image.shape
        assert denoised.dtype == np.uint8
    
    def test_adaptive_binarize(self):
        """Test adaptive binarization."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        binary = preprocess.adaptive_binarize(image, "adaptive_gaussian")
        
        assert binary.shape == image.shape
        assert binary.dtype == np.uint8
        # Should be binary (only 0 and 255 values)
        assert np.all(np.isin(binary, [0, 255]))


class TestPreprocessingPipeline:
    """Test the complete preprocessing pipeline."""
    
    def test_preprocess_image_standard_mode(self):
        """Test standard preprocessing mode."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        processed, info = preprocess.preprocess_image(
            image, 
            mode=preprocess.PreprocessingMode.STANDARD
        )
        
        assert isinstance(processed, np.ndarray)
        assert "original_shape" in info
        assert "mode" in info
        assert "applied_operations" in info
        assert "quality" in info
        assert "quality_metrics" in info
    
    def test_preprocess_image_handwriting_mode(self):
        """Test handwriting-specific preprocessing mode."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        processed, info = preprocess.preprocess_image(
            image, 
            mode=preprocess.PreprocessingMode.HANDWRITING
        )
        
        assert isinstance(processed, np.ndarray)
        assert info["mode"] == "handwriting"
    
    def test_preprocess_image_printed_mode(self):
        """Test printed text preprocessing mode."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        processed, info = preprocess.preprocess_image(
            image, 
            mode=preprocess.PreprocessingMode.PRINTED
        )
        
        assert isinstance(processed, np.ndarray)
        assert info["mode"] == "printed"
    
    def test_preprocess_image_mixed_mode(self):
        """Test mixed content preprocessing mode."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        processed, info = preprocess.preprocess_image(
            image, 
            mode=preprocess.PreprocessingMode.MIXED
        )
        
        assert isinstance(processed, np.ndarray)
        assert info["mode"] == "mixed"


class TestPreprocessingModes:
    """Test preprocessing mode enums."""
    
    def test_image_quality_enum(self):
        """Test ImageQuality enum values."""
        assert preprocess.ImageQuality.EXCELLENT.value == "excellent"
        assert preprocess.ImageQuality.GOOD.value == "good"
        assert preprocess.ImageQuality.FAIR.value == "fair"
        assert preprocess.ImageQuality.POOR.value == "poor"
    
    def test_preprocessing_mode_enum(self):
        """Test PreprocessingMode enum values."""
        assert preprocess.PreprocessingMode.STANDARD.value == "standard"
        assert preprocess.PreprocessingMode.HANDWRITING.value == "handwriting"
        assert preprocess.PreprocessingMode.PRINTED.value == "printed"
        assert preprocess.PreprocessingMode.MIXED.value == "mixed"


if __name__ == "__main__":
    pytest.main([__file__])