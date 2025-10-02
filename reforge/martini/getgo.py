"""Gets contact maps for Go-Martini from RCSU server

Description:
    This module automates the download and extraction of Go maps from 
    http://info.ifpan.edu.pl/~rcsu/rcsu/index.html using Selenium and various 
    WebDriver managers. It checks for installed browsers, installs the appropriate 
    WebDriver if needed, and then processes a list of PDB files.

Author: DY
"""

import argparse
import os
import platform
import shutil
import subprocess as sp
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
import logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automate downloading and extracting Go maps from PDB files."
    )
    parser.add_argument(
        "-d", "--directory", required=True, help="Directory containing PDB files."
    )
    parser.add_argument("-f", "--file", required=True, help="Name of the PDB file.")
    return parser.parse_args()


def check_browser(browser_name, command):
    """Check if a browser is installed by looking for its executable command."""
    return shutil.which(command) is not None


def check_debian_package(package_name):
    """Check if a Debian package is installed."""
    try:
        result = sp.run(
            ["dpkg", "-s", package_name],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.returncode == 0
    except (sp.CalledProcessError, FileNotFoundError):
        return False


def install_webdriver(browser_name):
    """Install the WebDriver for the specified browser."""
    try:
        if browser_name == "chrome":
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service as ChromeService

            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service)
        elif browser_name == "firefox":
            # Import once at module level; avoid redefinition warnings by aliasing
            service = webdriver.firefox.service.Service(
                GeckoDriverManager().install()
            )
            driver = webdriver.Firefox(service=service)
        elif browser_name == "edge":
            from webdriver_manager.microsoft import EdgeChromiumDriverManager
            from selenium.webdriver.edge.service import Service as EdgeService

            service = EdgeService(EdgeChromiumDriverManager().install())
            driver = webdriver.Edge(service=service)
        elif browser_name == "safari" and platform.system() == "Darwin":
            driver = webdriver.Safari()
        else:
            logger.info(
                "%s: WebDriver installation is not supported in this script.",
                browser_name.capitalize(),
            )
            return None

        logger.info("%s: WebDriver installed successfully.", browser_name.capitalize())
        driver.quit()
    except Exception as e:
        logger.info(
            "Error installing WebDriver for %s: %s", browser_name.capitalize(), str(e)
        )


def check_browsers():
    """Check for installed browsers and log their status."""
    browsers = {
        "firefox": check_browser("Firefox", "firefox")
        or check_debian_package("firefox-esr"),
        "chrome": check_browser("Chrome", "google-chrome")
        or check_browser("Chromium", "chromium-browser"),
        "edge": check_browser("Edge", "microsoft-edge"),
        "safari": platform.system() == "Darwin" and check_browser("Safari", "safari"),
    }

    logger.info("Installed browsers:")
    for browser, installed in browsers.items():
        logger.info("%s: %s", browser.capitalize(), "Installed" if installed else "Not installed")


def check_geckodriver_installed():
    """Check if geckodriver is installed by running '--version'."""
    try:
        sp.run(["geckodriver", "--version"], capture_output=True, text=True, check=True)
        return True
    except (sp.CalledProcessError, FileNotFoundError):
        return False


def init_webdriver(download_dir):
    """
    Initialize the Firefox WebDriver with download preferences.
    
    Args:
        download_dir (str): Directory to download files to.
    
    Returns:
        webdriver.Firefox: The initialized Firefox WebDriver.
    """
    logger.info("Initializing WebDriver...")
    options = Options()
    options.add_argument("-headless")
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.download.dir", os.path.abspath(download_dir))
    options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/x-gzip")
    if check_geckodriver_installed():
        driver = webdriver.Firefox(options=options)
    else:
        driver = webdriver.Firefox(
            service=webdriver.firefox.service.Service(GeckoDriverManager().install()),
            options=options,
        )
    logger.info("WebDriver initialized.")
    return driver


def get_go_maps(driver, pdb_files):
    """
    Use Selenium to automate downloading Go maps from the server.
    
    Args:
        driver: Selenium WebDriver.
        pdb_files (list): List of PDB file paths.
    """
    logger.info("Submitting PDBs...")
    for pdb_file in pdb_files:
        logger.info("Processing %s...", pdb_file)
        driver.get("http://info.ifpan.edu.pl/~rcsu/rcsu/index.html")
        try:
            pdb_input = driver.find_element(By.NAME, "filename")
            pdb_input.send_keys(pdb_file)
            driver.find_element(By.XPATH, "//input[@type='SUBMIT']").click()
            download_link = WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, "here"))
            )
            logger.info("Downloading the map...")
            download_link.click()
            time.sleep(10)  # Wait for download to complete
            logger.info("Downloaded Go map for %s.", pdb_file)
        except Exception as e:
            logger.info("Error processing %s: %s", pdb_file, str(e))
    driver.quit()
    logger.info("Go maps download process completed.")


def extract_go_maps(wdir):
    """
    Extract downloaded Go maps from tar.gz files and organize them.
    
    Args:
        wdir (str): Working directory containing the downloads.
    """
    logger.info("Extracting Go maps...")
    tgz_files = [f for f in os.listdir(wdir) if f.endswith(".tgz")]
    for f in tgz_files:
        tgz_path = os.path.join(wdir, f)
        logger.info("Extracting %s...", f)
        sp.run(["tar", "-xzf", tgz_path, "-C", wdir], check=True)
        os.remove(tgz_path)
    work2_dir = os.path.join(wdir, "work2")
    if os.path.exists(work2_dir):
        dirs = list(os.listdir(work2_dir))
        for d in dirs:
            source_dir = os.path.join(work2_dir, d)
            files = os.listdir(source_dir)
            if files:
                source_file = os.path.join(source_dir, files[0])
                shutil.move(source_file, os.path.join(wdir, files[0]))
        shutil.rmtree(work2_dir)
        logger.info("Extraction complete. Maps are in %s.", os.path.abspath(wdir))
    else:
        logger.info("Directory %s not found", work2_dir)


def get_go(wdir, path_to_pdbs):
    """
    Check for browsers, initialize the WebDriver, download, and extract Go maps.
    
    Args:
        wdir (str): Working directory for downloads.
        path_to_pdbs (list): List of PDB file paths to process.
    """
    check_browsers()
    driver = init_webdriver(wdir)
    get_go_maps(driver, path_to_pdbs)
    extract_go_maps(wdir)


if __name__ == "__main__":
    args = parse_arguments()
    WDIR = args.directory
    PDB_FILE = args.file
    DOWNLOADS_ABSPATH = os.path.abspath(WDIR)

    check_browsers()
    # Initialize driver with download directory set to WDIR
    driver = init_webdriver(WDIR)
    # Here we pass a list containing the single PDB file;
    # adjust as needed for multiple files.
    get_go_maps(driver, [PDB_FILE])
    extract_go_maps(WDIR)
