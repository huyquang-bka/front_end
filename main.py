import streamlit as st
from OCR import *
from Plate import *


st.title("automatic license plate recognition".upper())
st.sidebar.title("SETTINGS")
st.sidebar.markdown("-" * 20)
st.sidebar.markdown("Upload license plate image  here!".upper())
file_uploader = st.sidebar.file_uploader("", ["jpg", "png", "jpeg", "gif"])

if file_uploader is not None:
    image = file_uploader.read()
    with open(f"Data/{file_uploader.name}", "wb+") as f:
        f.write(image)
    img = cv2.imread(f"Data/{file_uploader.name}")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.sidebar.image(img)
    st.image(img, use_column_width=True)
    try:
        status, crop = plate_detection(img_copy)
        st.image(crop, use_column_width=True)
        st.markdown("-" * 20)
        block_list_check = st.checkbox("Block list detection (Number + Alphaber UpperCase + special characters)")
        if block_list_check:
            vn = st.checkbox("Use Vietnamese License Plate")
            if vn:
                default_list = ["I", "J", "O", "Q", "W", "R"]
                label = "In Vietnamese, On civilian license plates, the letters I, J, O, Q, W, R are not used"
            else:
                default_list = None
                label = ""
            block_list = st.multiselect(label=label,
                                        options=[char for char in (string.ascii_uppercase + string.digits)],
                                        default=default_list)
        else:
            block_list = None

        allow_list = process_allowlist(block_list)
        lp_text = st.button("LP to text")
        if lp_text:
            thre_mor, text = process_image_chracter(crop)
            st.image(thre_mor, use_column_width=True)
            st.markdown(f"LP Number: **{text}**")
    except:
        st.sidebar.markdown("**CAN'T FIND License Plate**".upper(), unsafe_allow_html=True)
        st.markdown("**CAN'T FIND License Plate**".upper(), unsafe_allow_html=True)
