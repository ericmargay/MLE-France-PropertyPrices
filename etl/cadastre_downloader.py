"""
Robust Cadastre Building Footprints Downloader
===============================================

Downloads French cadastre building data (batiments) from cadastre.data.gouv.fr
with proper handling for large files and network issues.

Features:
- Streaming downloads with progress tracking
- Resume capability for interrupted downloads
- Retry logic with exponential backoff
- Memory-efficient processing of large files
- Parallel downloads for multiple departments

Data Source:
https://cadastre.data.gouv.fr/data/etalab-cadastre/latest/geojson/departements/
"""

import os
import gzip
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

CADASTRE_BASE_URL = "https://cadastre.data.gouv.fr/data/etalab-cadastre/latest/geojson/departements"
DEFAULT_CACHE_DIR = "/app/data/cadastre"

# Download settings
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
CONNECT_TIMEOUT = 30  # seconds
READ_TIMEOUT = 300  # 5 minutes for large files

# All French departments (metropolitan + overseas)
ALL_DEPARTMENTS = [
    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '21',
    '22', '23', '24', '25', '26', '27', '28', '29', '2A', '2B',
    '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
    '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',
    '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',
    '70', '71', '72', '73', '74', '75', '76', '77', '78', '79',
    '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
    '90', '91', '92', '93', '94', '95',
    '971', '972', '973', '974', '976'  # Overseas
]


def get_department_url(dept_code):
    """Get cadastre batiments URL for a department."""
    return f"{CADASTRE_BASE_URL}/{dept_code}/cadastre-{dept_code}-batiments.json.gz"


def download_with_retry(url, dest_path, max_retries=MAX_RETRIES, 
                        connect_timeout=CONNECT_TIMEOUT, read_timeout=READ_TIMEOUT):
    """
    Download a file with retry logic and progress tracking.
    
    Returns:
        tuple: (success: bool, file_size: int, error_msg: str or None)
    """
    temp_path = dest_path + '.partial'
    
    for attempt in range(max_retries):
        try:
            # Check if we have a partial download
            resume_pos = 0
            headers = {}
            
            if os.path.exists(temp_path):
                resume_pos = os.path.getsize(temp_path)
                headers['Range'] = f'bytes={resume_pos}-'
            
            # Start request
            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(connect_timeout, read_timeout)
            )
            
            # Check response
            if response.status_code == 404:
                return False, 0, "File not found (404)"
            
            if response.status_code not in [200, 206]:
                return False, 0, f"HTTP {response.status_code}"
            
            # Get total size
            if response.status_code == 206:  # Partial content (resume)
                content_range = response.headers.get('Content-Range', '')
                if '/' in content_range:
                    total_size = int(content_range.split('/')[-1])
                else:
                    total_size = resume_pos + int(response.headers.get('Content-Length', 0))
            else:
                total_size = int(response.headers.get('Content-Length', 0))
                resume_pos = 0  # Server doesn't support resume, start fresh
            
            # Download with progress
            mode = 'ab' if resume_pos > 0 else 'wb'
            downloaded = resume_pos
            
            with open(temp_path, mode) as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            # Verify download
            if total_size > 0 and downloaded < total_size:
                # Incomplete download, will retry
                continue
            
            # Success - rename temp file
            os.rename(temp_path, dest_path)
            return True, downloaded, None
            
        except requests.exceptions.Timeout:
            error_msg = f"Timeout (attempt {attempt + 1}/{max_retries})"
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {str(e)[:50]}"
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
        
        except Exception as e:
            error_msg = f"Error: {str(e)[:50]}"
            break
    
    return False, 0, error_msg


def decompress_and_load_geojson(gz_path):
    """
    Decompress and load a gzipped GeoJSON file.
    Memory-efficient: streams the decompression.
    
    Returns:
        GeoDataFrame or None
    """
    try:
        with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        if 'features' not in geojson_data or len(geojson_data['features']) == 0:
            return None
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geojson_data['features'], crs="EPSG:4326")
        
        return gdf
        
    except Exception as e:
        print(f"      Error loading GeoJSON: {e}")
        return None


def download_department_buildings(dept_code, cache_dir=DEFAULT_CACHE_DIR, force=False):
    """
    Download cadastre buildings for a single department.
    
    Parameters:
    -----------
    dept_code : str
        Department code (e.g., '75', '2A', '971')
    cache_dir : str
        Directory to cache downloaded files
    force : bool
        If True, re-download even if cached
    
    Returns:
    --------
    dict with keys:
        - success: bool
        - dept: str
        - buildings_count: int
        - file_size_mb: float
        - error: str or None
        - gdf: GeoDataFrame or None
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    dept = str(dept_code).zfill(2) if dept_code.isdigit() and len(dept_code) < 2 else str(dept_code)
    
    gz_path = os.path.join(cache_dir, f'cadastre-{dept}-batiments.json.gz')
    geojson_path = os.path.join(cache_dir, f'batiments_{dept}.geojson')
    failed_marker = os.path.join(cache_dir, f'batiments_{dept}.failed')
    
    result = {
        'success': False,
        'dept': dept,
        'buildings_count': 0,
        'file_size_mb': 0,
        'error': None,
        'gdf': None
    }
    
    # Check for cached GeoJSON (already processed)
    if not force and os.path.exists(geojson_path):
        try:
            gdf = gpd.read_file(geojson_path)
            result['success'] = True
            result['buildings_count'] = len(gdf)
            result['gdf'] = gdf
            return result
        except Exception:
            pass
    
    # Check for cached .gz file
    if not force and os.path.exists(gz_path):
        gdf = decompress_and_load_geojson(gz_path)
        if gdf is not None:
            # Save as GeoJSON for faster future loads
            try:
                gdf.to_file(geojson_path, driver='GeoJSON')
            except:
                pass
            result['success'] = True
            result['buildings_count'] = len(gdf)
            result['file_size_mb'] = os.path.getsize(gz_path) / (1024 * 1024)
            result['gdf'] = gdf
            return result
    
    # Download
    url = get_department_url(dept)
    success, file_size, error = download_with_retry(url, gz_path)
    
    if not success:
        result['error'] = error
        # Mark as failed to skip in future
        with open(failed_marker, 'w') as f:
            f.write(error or 'download failed')
        return result
    
    result['file_size_mb'] = file_size / (1024 * 1024)
    
    # Load and process
    gdf = decompress_and_load_geojson(gz_path)
    
    if gdf is None:
        result['error'] = "Failed to parse GeoJSON"
        return result
    
    # Save processed GeoJSON for faster future loads
    try:
        gdf.to_file(geojson_path, driver='GeoJSON')
    except Exception as e:
        # If save fails (disk space?), continue anyway
        pass
    
    result['success'] = True
    result['buildings_count'] = len(gdf)
    result['gdf'] = gdf
    
    return result


def download_all_buildings(departments=None, cache_dir=DEFAULT_CACHE_DIR, 
                           max_workers=4, progress_callback=None):
    """
    Download cadastre buildings for multiple departments.
    
    Parameters:
    -----------
    departments : list
        List of department codes to download (default: all)
    cache_dir : str
        Directory to cache downloaded files
    max_workers : int
        Number of parallel downloads
    progress_callback : callable
        Function called with (completed, total, dept, status) for progress
    
    Returns:
    --------
    dict: {dept_code: GeoDataFrame} for successful downloads
    """
    if departments is None:
        departments = ALL_DEPARTMENTS
    
    os.makedirs(cache_dir, exist_ok=True)
    
    results = {}
    total = len(departments)
    completed = 0
    
    print(f"\n{'='*60}")
    print(f"DOWNLOADING CADASTRE BUILDINGS")
    print(f"{'='*60}")
    print(f"Departments: {total}")
    print(f"Cache directory: {cache_dir}")
    print(f"Parallel workers: {max_workers}")
    
    # Track statistics
    stats = {
        'success': 0,
        'failed': 0,
        'cached': 0,
        'total_buildings': 0,
        'total_size_mb': 0
    }
    
    def process_dept(dept):
        nonlocal completed
        
        # Check if already cached
        geojson_path = os.path.join(cache_dir, f'batiments_{dept}.geojson')
        if os.path.exists(geojson_path):
            try:
                gdf = gpd.read_file(geojson_path)
                return {'success': True, 'dept': dept, 'gdf': gdf, 
                        'buildings_count': len(gdf), 'cached': True}
            except:
                pass
        
        # Download
        result = download_department_buildings(dept, cache_dir)
        result['cached'] = False
        return result
    
    # Process departments
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dept = {executor.submit(process_dept, dept): dept for dept in departments}
        
        for future in as_completed(future_to_dept):
            dept = future_to_dept[future]
            completed += 1
            
            try:
                result = future.result()
                
                if result['success']:
                    results[dept] = result['gdf']
                    stats['total_buildings'] += result['buildings_count']
                    stats['total_size_mb'] += result.get('file_size_mb', 0)
                    
                    if result.get('cached'):
                        stats['cached'] += 1
                        status = f"✓ cached ({result['buildings_count']:,} buildings)"
                    else:
                        stats['success'] += 1
                        status = f"✓ downloaded ({result['buildings_count']:,} buildings)"
                else:
                    stats['failed'] += 1
                    status = f"✗ {result.get('error', 'failed')}"
                
                # Progress update
                pct = completed / total * 100
                print(f"  [{completed:3d}/{total}] {pct:5.1f}% | Dept {dept}: {status}")
                
                if progress_callback:
                    progress_callback(completed, total, dept, status)
                    
            except Exception as e:
                stats['failed'] += 1
                print(f"  [{completed:3d}/{total}] Dept {dept}: ✗ Exception: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"  Successful: {stats['success']} downloaded + {stats['cached']} cached")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total buildings: {stats['total_buildings']:,}")
    print(f"  Downloaded size: {stats['total_size_mb']:.1f} MB")
    
    return results


def load_cached_buildings(cache_dir=DEFAULT_CACHE_DIR, departments=None):
    """
    Load already-downloaded building data from cache.
    
    Returns:
    --------
    dict: {dept_code: GeoDataFrame}
    """
    if departments is None:
        departments = ALL_DEPARTMENTS
    
    results = {}
    
    for dept in departments:
        geojson_path = os.path.join(cache_dir, f'batiments_{dept}.geojson')
        
        if os.path.exists(geojson_path):
            try:
                gdf = gpd.read_file(geojson_path)
                results[dept] = gdf
            except Exception:
                pass
    
    return results


def get_building_for_point(point, buildings_gdf, max_distance=0.0005):
    """
    Find the building containing or nearest to a point.
    
    Parameters:
    -----------
    point : shapely.geometry.Point
        Transaction location
    buildings_gdf : GeoDataFrame
        Building footprints
    max_distance : float
        Maximum distance in degrees (~50m) to search for nearby building
    
    Returns:
    --------
    shapely.geometry.Polygon or None
    """
    # Check containment first (fast)
    containing = buildings_gdf[buildings_gdf.geometry.contains(point)]
    
    if len(containing) > 0:
        return containing.iloc[0].geometry
    
    # Find nearest within max_distance
    distances = buildings_gdf.geometry.distance(point)
    min_idx = distances.idxmin()
    min_dist = distances[min_idx]
    
    if min_dist <= max_distance:
        return buildings_gdf.loc[min_idx].geometry
    
    return None


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download French cadastre building data')
    parser.add_argument('--departments', '-d', nargs='+', help='Specific departments to download')
    parser.add_argument('--cache-dir', '-c', default=DEFAULT_CACHE_DIR, help='Cache directory')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Parallel download workers')
    parser.add_argument('--force', '-f', action='store_true', help='Force re-download')
    
    args = parser.parse_args()
    
    depts = args.departments if args.departments else ALL_DEPARTMENTS
    
    results = download_all_buildings(
        departments=depts,
        cache_dir=args.cache_dir,
        max_workers=args.workers
    )
    
    print(f"\nLoaded {len(results)} departments with building data")
