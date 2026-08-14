import streamlit as st
import pandas as pd
from datetime import datetime
import re

# Page Config
st.set_page_config(page_title="AI Smoke Shop POS", layout="wide", page_icon="🚬")

# --- INITIALIZE MOCK INVENTORY & CART IN SESSION STATE ---
if "inventory" not in st.session_state:
    st.session_state.inventory = pd.DataFrame([
        {"SKU": "1001", "Item": "Geek Bar Pulse 15k - Blow Pop", "Price": 22.99, "Stock": 45, "Age_Restricted": True},
        {"SKU": "1002", "Item": "RAW Classic King Size Papers", "Price": 2.50, "Stock": 120, "Age_Restricted": True},
        {"SKU": "1003", "Item": "10-inch Beaker Glass Water Pipe", "Price": 49.99, "Stock": 8, "Age_Restricted": True},
        {"SKU": "1004", "Item": "Torching Lighter Dual-Jet", "Price": 14.99, "Stock": 22, "Age_Restricted": False},
    ])

if "cart" not in st.session_state:
    st.session_state.cart = []

if "id_verified" not in st.session_state:
    st.session_state.id_verified = False

if "customer_age" not in st.session_state:
    st.session_state.customer_age = None

# --- AAMVA PDF417 BARCODE PARSER ---
def parse_aamva_dl(barcode_data: str):
    """
    Parses AAMVA standard driver's license PDF417 raw text string.
    Extracts Date of Birth (DOB) and Expiration Date.
    """
    try:
        # Standard AAMVA tag for Date of Birth is DBB
        dob_match = re.search(r'DBB(\d{8})', barcode_data)
        
        if dob_match:
            dob_str = dob_match.group(1)
            
            # Format handling: MMDDYYYY vs YYYYMMDD
            if int(dob_str[:2]) <= 12 and int(dob_str[2:4]) <= 31:
                dob = datetime.strptime(dob_str, "%m%d%Y")
            else:
                dob = datetime.strptime(dob_str, "%Y%m%d")
                
            today = datetime.now()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return True, age, dob.strftime("%m/%d/%Y")
        else:
            return False, 0, "DOB Field Not Found"
    except Exception as e:
        return False, 0, f"Parse Error: {str(e)}"

# --- HEADER & UI LAYOUT ---
st.title("🚬 Smoke Shop Smart POS")
col_pos, col_id = st.columns([2, 1])

# --- RIGHT SIDE: ID AGE VERIFICATION SYSTEM ---
with col_id:
    st.header("🆔 ID Verification")
    
    # Text input acts as the scanner receiver (hardware scanners type into active fields and press Enter)
    scan_input = st.text_input("Scan DL Barcode Here (Active Field):", type="password", key="scanner_input")
    
    if scan_input:
        success, age, dob_fmt = parse_aamva_dl(scan_input)
        if success:
            st.session_state.customer_age = age
            if age >= 21:
                st.session_state.id_verified = True
                st.success(f"✅ AGE VERIFIED: {age} Years Old (DOB: {dob_fmt})")
            else:
                st.session_state.id_verified = False
                st.error(f"🚨 UNDERAGE CUSTOMER: {age} Years Old! DO NOT SELL.")
        else:
            st.warning("⚠️ Could not parse barcode format. Re-scan or check scanner configuration.")
            
    # Status Banner
    if st.session_state.id_verified:
        st.info(f"Verified Customer Active | Age: {st.session_state.customer_age}")
    else:
        st.error("🔒 Age Restricted Sale Locked (Scan ID to Unlock Checkout)")

    if st.button("Reset / Clear Customer"):
        st.session_state.id_verified = False
        st.session_state.customer_age = None
        st.rerun()

# --- LEFT SIDE: CART & INVENTORY ---
with col_pos:
    st.header("🛒 Checkout")
    
    # SKU Quick Add
    sku_col, qty_col = st.columns([3, 1])
    with sku_col:
        selected_sku = st.selectbox("Select/Scan Product:", st.session_state.inventory["SKU"] + " - " + st.session_state.inventory["Item"])
    
    if st.button("Add to Cart"):
        sku_id = selected_sku.split(" - ")[0]
        item_data = st.session_state.inventory[st.session_state.inventory["SKU"] == sku_id].iloc[0]
        st.session_state.cart.append(item_data.to_dict())
        st.toast(f"Added {item_data['Item']} to cart")

    # Display Cart
    if st.session_state.cart:
        cart_df = pd.DataFrame(st.session_state.cart)
        st.dataframe(cart_df[["Item", "Price", "Age_Restricted"]], use_container_width=True)
        
        total_price = cart_df["Price"].sum()
        st.markdown(f"### Total: **${total_price:.2f}**")
        
        has_age_restricted = any(cart_df["Age_Restricted"])
        
        # Checkout Safeguard Logic
        if has_age_restricted and not st.session_state.id_verified:
            st.button("💳 COMPLETE SALE", disabled=True, help="Scan a 21+ Driver's License to enable checkout.")
        else:
            if st.button("💳 COMPLETE SALE", type="primary"):
                st.balloons()
                st.success(f"Sale Complete! Total Charged: ${total_price:.2f}")
                st.session_state.cart = []
                st.session_state.id_verified = False
                st.rerun()
    else:
        st.write("Cart is currently empty.")
