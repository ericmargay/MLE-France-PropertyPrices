"""
Data download utilities for French property prices project

Downloads:
1. DVF (Demandes de Valeurs Foncières) transaction data
2. French administrative boundary GeoJSON files
"""

import requests
import os
from pathlib import Path
import time

# Directories
DATA_DIR = Path('/app/data')
RAW_DIR = DATA_DIR / 'raw'
GEOMETRY_DIR = DATA_DIR / 'geometries'

# Create directories if they don't exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url, filepath, chunk_size=8192):
    """
    Download a file with progress indicator
    
    Parameters:
    -----------
    url : str
        URL to download from
    filepath : Path
        Destination file path
    chunk_size : int
        Download chunk size in bytes
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress indicator
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f'\r  Progress: {percent:.1f}% ({downloaded / 1024**2:.1f} MB)', end='')
        
        print()  # New line after progress
        file_size = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ Downloaded: {filepath.name} ({file_size:.2f} MB)")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading {url}: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove partial download
        raise


def download_dvf_data(year=2023):
    """
    Download DVF property transaction data for a specific year
    
    Parameters:
    -----------
    year : int
        Year to download (default: 2023)
    """
    print(f"\n{'='*60}")
    print(f"DOWNLOADING DVF DATA FOR {year}")
    print(f"{'='*60}\n")
    
    filename = f'dvf_{year}.csv.gz'
    filepath = RAW_DIR / filename
    
    if filepath.exists():
        file_size = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ {filename} already exists ({file_size:.2f} MB)")
        return filepath
    
    # DVF data URL
    url = f"https://files.data.gouv.fr/geo-dvf/latest/csv/{year}/full.csv.gz"
    
    print(f"Downloading from: {url}")
    print("This may take several minutes (~100 MB)...")
    
    download_file(url, filepath)
    
    return filepath


def download_geometries():
    """
    Download French administrative boundary GeoJSON files
    Uses reliable GitHub-hosted GeoJSON files
    """
    print(f"\n{'='*60}")
    print("DOWNLOADING GEOMETRY FILES")
    print(f"{'='*60}\n")
    
    # Using reliable GitHub repository with French boundaries
    base_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master"
    
    files = {
        'regions': {
            'url': f'{base_url}/regions.geojson',
            'expected_size': 5  # MB (approximate)
        },
        'departements': {
            'url': f'{base_url}/departements.geojson',
            'expected_size': 15  # MB
        },
        'communes': {
            'url': f'{base_url}/communes.geojson',
            'expected_size': 180  # MB (large file!)
        }
    }
    
    for name, info in files.items():
        filepath = GEOMETRY_DIR / f'{name}.geojson'
        
        if filepath.exists():
            file_size = filepath.stat().st_size / (1024 * 1024)
            
            # Check if file is corrupted (too small)
            if file_size < 0.1:  # Less than 100 KB = probably corrupted
                print(f"⚠ {name}.geojson exists but is corrupted ({file_size:.2f} MB)")
                print(f"  Removing and re-downloading...")
                filepath.unlink()
            else:
                print(f"✓ {name}.geojson already exists ({file_size:.2f} MB)")
                continue
        
        print(f"\nDownloading {name}.geojson...")
        print(f"  Expected size: ~{info['expected_size']} MB")
        
        try:
            download_file(info['url'], filepath)
            
            # Verify download
            file_size = filepath.stat().st_size / (1024 * 1024)
            if file_size < 0.1:
                raise ValueError(f"Downloaded file is too small ({file_size:.2f} MB)")
            
            # Verify it's valid JSON (not HTML error page)
            with open(filepath, 'r') as f:
                first_chars = f.read(100)
                if first_chars.strip().startswith('<'):
                    raise ValueError("File contains HTML, not GeoJSON")
                if not (first_chars.strip().startswith('{') or first_chars.strip().startswith('[')):
                    raise ValueError("File is not valid JSON")
            
            print(f"  ✓ Verified: Valid GeoJSON")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            if filepath.exists():
                filepath.unlink()
            raise
    
    print(f"\n✓ All geometry files downloaded and verified")


def verify_downloads():
    """
    Verify all required files are downloaded
    
    Returns:
    --------
    bool : True if all files exist and are valid
    """
    print(f"\n{'='*60}")
    print("VERIFYING DOWNLOADS")
    print(f"{'='*60}\n")
    
    required_files = {
        RAW_DIR / 'dvf_2023.csv.gz': 50,  # Minimum 50 MB
        GEOMETRY_DIR / 'regions.geojson': 1,  # Minimum 1 MB
        GEOMETRY_DIR / 'departements.geojson': 5,  # Minimum 5 MB
        GEOMETRY_DIR / 'communes.geojson': 100,  # Minimum 100 MB
    }
    
    all_exist = True
    
    for filepath, min_size_mb in required_files.items():
        if filepath.exists():
            file_size = filepath.stat().st_size / (1024 * 1024)
            
            if file_size < min_size_mb:
                print(f"⚠ {filepath.name:30} ({file_size:.2f} MB) - TOO SMALL! Expected >{min_size_mb} MB")
                all_exist = False
            else:
                print(f"✓ {filepath.name:30} ({file_size:.2f} MB)")
        else:
            print(f"✗ {filepath.name:30} NOT FOUND")
            all_exist = False
    
    return all_exist


def main():
    """
    Main download function
    """
    print("\n" + "="*60)
    print("DATA DOWNLOAD SCRIPT")
    print("="*60)
    
    try:
        # Check if already downloaded
        if verify_downloads():
            print("\n✓ All files already downloaded and verified")
            return True
        
        # Download DVF data
        download_dvf_data(2023)
        
        # Download geometries
        download_geometries()
        
        # Final verification
        if verify_downloads():
            print("\n" + "="*60)
            print("✓ ALL DOWNLOADS COMPLETE")
            print("="*60)
            return True
        else:
            print("\n✗ Some files are missing or corrupted")
            return False
            
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)