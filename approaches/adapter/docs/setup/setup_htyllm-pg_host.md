## How to Get to htyllm-pg Machine for Development

This explains how to access the `htyllm-pg` machine for development purposes.

**1. Generate and Add SSH Key:**

   - **On your local machine:** Open your terminal and generate a new SSH key pair:
     ```bash
     ssh-keygen
     ```
     This will create a public and private key in `~/.ssh/`.

   - **Add the public key to GitLab:**
     - Go to your university's GitLab instance.
     - Click on your **Profile** icon.
     - Select **Edit profile**.
     - Navigate to **SSH Keys** in the sidebar.
     - Click **Add new key**.
     - Copy the content of your **public key file** (e.g., `~/.ssh/<your_keyfilename>.pub`) and paste it into the "Key" field.
     - Click **Add key**.

**2. Connect to the Gateway Server:**

   - Open your terminal and connect to the SSH gateway:
     ```bash
     ssh <your_uni_username>@sshgate.cs.upb.de
     ```
     Type in your uni password, if you are asked for it

**3. Connect to the Target Machine:**

   - Once logged into `sshgate.cs.upb.de`, connect to the `htyllm-pg` host:
     ```bash
     ssh <your_uni_username>@htyllm-pg.cs.uni-paderborn.de
     ```
   - You might be asked for your university password.

   - You should then be logged into `htyllm-pg.cs.uni-paderborn.de`. 
   
The machine has 32 cores and approximately 1TB of storage. For large datasets, use the `/data` directory. You should also have `sudo` rights.

## (Optional) Convenient Remote Development with VS Code

This enables you to directly login into the @htyllm-pg.cs.uni-paderborn.de host without the intermediate step of logging in into @sshgate.cs.upb.de
This is required if you want to develop remotely with vs code (as far as i know)

**### Set up Proxy Jump in SSH Configuration:**

   - **On your local machine:** Create or edit the SSH configuration file:
     ```
     ~/.ssh/config
     ```
   - Add the following content + adjust `User` and `IdentityFile`

     ```
     Host sshgate
         HostName sshgate.cs.upb.de
         User <your_uni_username>
         IdentityFile ~/.ssh/<your_private_key_file_name>  # Ensure this path is correct for your private key
         ForwardAgent yes

     Host htyllm-pg
         HostName htyllm-pg.cs.uni-paderborn.de
         User <your_uni_username>
         ProxyJump sshgate
         ForwardAgent yes
     ```

**### Set up VS Code Remote Development:**

   1. Install VS Code if you dont have it
   2. Install "Remote - SSH" Extension: Open VS Code, go to the Extensions view, search for "Remote - SSH", install it.
   3. Connect to Host: After installed the extension,  press F1, type "Remote-SSH: Connect to Host...", and select the suggested command.
   4. Choose Target Host: VS Code should now recognize the `htyllm-pg` host from your SSH configuration. Select "htyllm-pg".
   5. Enter yor uni password when you are asked for it

You should then be connected to `htyllm-pg.cs.uni-paderborn.de` directly from VS Code. You can open your project folders and start developing. The integrated terminal in VS Code will also be directly connected to the `htyllm-pg` host.
I then suggest you to clone our repository, create a python venv, install necessary requiremetns. You should be able to develop then.



If you encounter any issues, feel free to reach out for help.
If there are other ways or shortcuts, so feel free to adapt this guide if you find better approaches.