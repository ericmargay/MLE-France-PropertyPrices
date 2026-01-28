#!/usr/bin/env python3
"""
Cadastre Building Footprints Downloader
=======================================

Standalone script to download French cadastre building data.
Run this BEFORE running the ETL pipeline.

Usage:
    docker-compose run --rm etl python download_cadastre.py
    docker-compose run --rm etl python download_cadastre.py --status
    docker-compose run --rm etl python download_cadastre.py --retry-failed
"""

import os
import sys
import json
import time
import gzip
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
CADASTRE_BASE_URL = "https://cadastre.data.gouv.fr/data/etalab-cadastre/latest/geojson/departements"
CACHE_DIR = "/app/data/cadastre"

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
    '90', '91', '92', '93', '94', '95'
]

MAX_RETRIES = 5
INITIAL_TIMEOUT = 120
MAX_TIMEOUT = 600
CHUNK_SIZE = 1024 * 1024


def get_download_url(dept):
    return f"{CADASTRE_BASE_URL}/{dept}/cadastre-{dept}-batiments.json.gz"


def check_existing_files(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    status = {}
    
    for dept in ALL_DEPARTMENTS:
        geojson_file = os.path.join(cache_dir, f'batiments_{dept}.geojson')
        sampled_file = os.path.join(cache_dir, f'batiments_{dept}_sampled.geojson')
        gz_file = os.path.join(cache_dir, f'cadastre-{dept}-batiments.json.gz')
        failed_file = os.path.join(cache_dir, f'batiments_{dept}.failed')
        
        if os.path.exists(geojson_file) and os.path.getsize(geojson_file) > 1000:
            status[dept] = {'status': 'ok', 'file': geojson_file, 'size': os.path.getsize(geojson_file)}
        elif os.path.exists(sampled_file) and os.path.getsize(sampled_file) > 1000:
            status[dept] = {'status': 'ok', 'file': sampled_file, 'size': os.path.getsize(sampled_file)}
        elif os.path.exists(gz_file) and os.path.getsize(gz_file) > 1000:
            status[dept] = {'status': 'ok', 'file': gz_file, 'size': os.path.getsize(gz_file)}
        elif os.path.exists(failed_file):
            status[dept] = {'status': 'failed', 'file': failed_file, 'size': 0}
        else:
            status[dept] = {'status': 'missing', 'file': None, 'size': 0}
    
    return status


def download_department(dept, cache_dir, timeout=INITIAL_TIMEOUT):
    url = get_download_url(dept)
    gz_path = os.path.join(cache_dir, f'cadastre-{dept}-batiments.json.gz')
    geojson_path = os.path.join(cache_dir, f'batiments_{dept}.geojson')
    failed_path = os.path.join(cache_dir, f'batiments_{dept}.failed')
    
    if os.path.exists(failed_path):
        os.remove(failed_path)
    
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        
        if response.status_code == 404:
            return {'dept': dept, 'status': 'not_found', 'error': 'File not found'}
        
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        temp_path = gz_path + '.partial'
        downloaded = 0
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        if total_size > 0 and downloaded < total_size * 0.95:
            os.remove(temp_path)
            return {'dept': dept, 'status': 'incomplete', 'error': f'Incomplete: {downloaded}/{total_size}'}
        
        os.rename(temp_path, gz_path)
        
        # Decompress
        try:
            with gzip.open(gz_path, 'rt', encoding='utf-8') as f_in:
                data = json.load(f_in)
            
            with open(geojson_path, 'w', encoding='utf-8') as f_out:
                json.dump(data, f_out)
            
            os.remove(gz_path)
            
            return {
                'dept': dept,
                'status': 'ok',
                'size': os.path.getsize(geojson_path),
                'features': len(data.get('features', []))
            }
        except Exception:
            return {
                'dept': dept,
                'status': 'ok',
                'size': os.path.getsize(gz_path),
                'features': 0,
                'note': 'Kept as .gz'
            }
    
    except requests.exceptions.Timeout:
        return {'dept': dept, 'status': 'timeout', 'error': f'Timeout {timeout}s'}
    except Exception as e:
        return {'dept': dept, 'status': 'error', 'error': str(e)[:100]}


def download_with_retry(dept, cache_dir, max_retries=MAX_RETRIES):
    timeout = INITIAL_TIMEOUT
    
    for attempt in range(max_retries):
        result = download_department(dept, cache_dir, timeout=timeout)
        
        if result['status'] == 'ok':
            return result
        
        if result['status'] == 'not_found':
            failed_path = os.path.join(cache_dir, f'batiments_{dept}.failed')
            with open(failed_path, 'w') as f:
                f.write('not_found')
            return result
        
        if attempt < max_retries - 1:
            timeout = min(timeout * 1.5, MAX_TIMEOUT)
            wait_time = 5 * (attempt + 1)
            print(f"    Retry {attempt + 2}/{max_retries} in {wait_time}s")
            time.sleep(wait_time)
    
    failed_path = os.path.join(cache_dir, f'batiments_{dept}.failed')
    with open(failed_path, 'w') as f:
        f.write(result.get('error', 'unknown')[:100])
    
    return result


def print_status(status):
    ok = [d for d, s in status.items() if s['status'] == 'ok']
    failed = [d for d, s in status.items() if s['status'] == 'failed']
    missing = [d for d, s in status.items() if s['status'] == 'missing']
    total_size = sum(s['size'] for s in status.values())
    
    print("\n" + "="*60)
    print("CADASTRE STATUS")
    print("="*60)
    print(f"  Downloaded: {len(ok)}/{len(ALL_DEPARTMENTS)}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Missing:    {len(missing)}")
    print(f"  Total size: {total_size / (1024**3):.1f} GB")
    
    if failed:
        print(f"\n  Failed: {', '.join(sorted(failed))}")
    if missing:
        print(f"\n  Missing: {', '.join(sorted(missing))}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--departments', '-d', type=str, default=None)
    parser.add_argument('--retry-failed', '-r', action='store_true')
    parser.add_argument('--status', '-s', action='store_true')
    parser.add_argument('--workers', '-w', type=int, default=2)
    parser.add_argument('--force', '-f', action='store_true')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CADASTRE DOWNLOADER")
    print("="*60)
    
    status = check_existing_files(CACHE_DIR)
    
    if args.status:
        print_status(status)
        return 0
    
    # Determine what to download
    if args.departments:
        to_download = [d.strip() for d in args.departments.split(',')]
    elif args.retry_failed:
        to_download = [d for d, s in status.items() if s['status'] in ('failed', 'missing')]
    elif args.force:
        to_download = ALL_DEPARTMENTS.copy()
    else:
        to_download = [d for d, s in status.items() if s['status'] == 'missing']
    
    if not args.force:
        to_download = [d for d in to_download if status.get(d, {}).get('status') != 'ok']
    
    if not to_download:
        print("\n All departments already downloaded!")
        print_status(status)
        return 0
    
    print(f"\nTo download: {len(to_download)} departments")
    print(f"Workers: {args.workers}")
    
    print("\n" + "-"*60)
    
    results = []
    completed = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_dept = {
            executor.submit(download_with_retry, dept, CACHE_DIR): dept 
            for dept in to_download
        }
        
        for future in as_completed(future_to_dept):
            dept = future_to_dept[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'ok':
                    size_mb = result.get('size', 0) / (1024*1024)
                    features = result.get('features', '?')
                    print(f"  [{completed:2d}/{len(to_download)}] {dept}: OK {size_mb:.1f} MB ({features} buildings)")
                else:
                    print(f"  [{completed:2d}/{len(to_download)}] {dept}: FAILED {result.get('error', '')}")
            except Exception as e:
                print(f"  [{completed:2d}/{len(to_download)}] {dept}: ERROR {e}")
                results.append({'dept': dept, 'status': 'error', 'error': str(e)})
    
    elapsed = time.time() - start_time
    
    status = check_existing_files(CACHE_DIR)
    print_status(status)
    print(f"\nTime: {elapsed/60:.1f} minutes")
    
    success = len([r for r in results if r['status'] == 'ok'])
    failed = len(results) - success
    
    if failed > 0:
        print(f"\n {failed} failed. Run --retry-failed")
        return 1
    
    print(f"\n All {success} downloaded!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
