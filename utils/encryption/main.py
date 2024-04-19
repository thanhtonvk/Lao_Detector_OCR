import random
import string
import json
import base64
from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Hash import SHA256, MD5
from Crypto.Signature import pkcs1_15
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import pad


def generate_secret_key():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=32))


def encrypt_to_base64(data, secret_key: str):
    # Convert data to bytes if it's not
    if isinstance(data, str):
        data = data.encode()
    # Create a new cipher using the secret key and a randomly generated IV
    cipher = AES.new(secret_key.encode(), AES.MODE_CBC)
    iv = cipher.iv
    # Encrypt and pad the data
    encrypted_data = cipher.encrypt(pad(data, AES.block_size))
    # Combine IV and encrypted data
    iv_and_encrypted_data = iv + encrypted_data
    # Encode to Base64 for easy transmission
    return base64.b64encode(iv_and_encrypted_data).decode("utf-8")


def decrypt_file_upload(encrypted_data, secret_key: str):
    # Extract the IV, which is the first 16 bytes of the file
    iv = encrypted_data[:16]
    encrypted_content = encrypted_data[16:]
    # Create a cipher object using the secret key and IV
    cipher = AES.new(secret_key.encode(), AES.MODE_CBC, iv)
    # Decrypt and return the content
    decrypted_content = cipher.decrypt(encrypted_content)
    return decrypted_content


def load_graph(path):
    with open(path, "rb") as f:
        protobuf_byte_str = f.read()
    return protobuf_byte_str


def encrypt_binary_string(raw, _key):
    bs = 32
    s = raw
    raw = s + str.encode((bs - len(s) % bs) * chr(bs - len(s) % bs))
    key = SHA256.new(_key.encode()).digest()
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return iv + cipher.encrypt(raw)


def decrypt_binary_string(enc, _key):
    key = SHA256.new(_key.encode()).digest()
    iv = enc[: AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    s = cipher.decrypt(enc[AES.block_size :])
    return s[: -ord(s[len(s) - 1 :])]


def save_encode_file(source_file, target_file, _key):
    binary_str = load_graph(source_file)
    binary_encode_str = encrypt_binary_string(binary_str, _key)
    with open(target_file, "wb") as f:
        f.write(binary_encode_str)


def load_encode_file(encode_file, _key):
    binary_str = load_graph(encode_file)
    binary_decode_str = decrypt_binary_string(binary_str, _key)
    return binary_decode_str


def generate_encryption_key(public_key: str):
    public_key_hash = SHA256.new(public_key.encode()).hexdigest()
    public_key_split = public_key.split("\n")
    combined_key_split = (
        public_key_split[: len(public_key_split) - 2]
        + [public_key_hash]
        + public_key_split[len(public_key_split) - 2 :]
    )
    combined_key = "\n".join(combined_key_split)
    combined_key_hash = MD5.new(combined_key.encode()).hexdigest()
    return combined_key_hash


def verify_base64_signature(public_key: str, signature: str, signature_data: dict):
    public_key_imported = RSA.import_key(public_key)
    signature_decode = base64.b64decode(signature)
    signature_message = json.dumps(signature_data, separators=(",", ":"))
    signature_message_hash = SHA256.new(signature_message.encode())
    try:
        pkcs1_15.new(public_key_imported).verify(
            signature_message_hash, signature_decode
        )
        return True
    except (ValueError, TypeError):
        return False


def verify_license(license_file: str, key_file: str):
    with open(key_file) as f:
        json_data = json.load(f)
    public_key = str(json_data["public_key"])
    with open(license_file) as f:
        json_data = json.load(f)
    license = {**json_data}
    verified = verify_base64_signature(
        public_key,
        license["key"],
        {
            key: value
            for key, value in license.items()
            if key not in ["key", "key_id"]
        },
    )
    if not verified:
        raise Exception("Invalid License")
    return license, public_key
