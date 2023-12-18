# RealTimeID
Real time interpolative decomposition for quantum impurity problems



## Environment Variable Setup

### Why Is This Necessary?
To ensure seamless functionality and portability of our project, we utilize the `REALTIMEID_PATH` environment variable. This approach dynamically configures the project's root directory, enabling consistent and correct module imports across various setups. It's a crucial step for maintaining the flexibility of the project across different systems and for simplifying collaborative development.

### Setting Up `REALTIMEID_PATH`

#### For macOS/Linux Users:
1. **Open Your Terminal**.
2. **Edit Your Shell Profile**: Depending on your shell, use one of the following commands:
   - For Bash: `nano ~/.bash_profile` or `nano ~/.bashrc`
   - For Zsh: `nano ~/.zshrc`
3. **Add the Environment Variable**:
   - Append the following line to the file:
     ```
     export REALTIMEID_PATH="/absolute/path/to/project"
     ```
   - Replace `/absolute/path/to/project` with the full path to your project's root directory.
4. **Save and Apply Changes**:
   - Save the file (`Ctrl + O`, `Enter` if using nano) and exit (`Ctrl + X` for nano).
   - Activate the changes with `source ~/.bash_profile`, `source ~/.bashrc`, or `source ~/.zshrc`.
5. **Verify the Setup**:
   - Confirm the variable is set: `echo $REALTIMEID_PATH`.

#### For Windows Users:
1. **Open System Properties**: Search for 'Environment Variables' and select "Edit the system environment variables."
2. **Create the Environment Variable**:
   - Click "Environment Variables."
   - Under "User variables," click "New."
   - Set "Variable name" as `REALTIMEID_PATH`.
   - Set "Variable value" as the full path to your project's root directory.
3. **Save and Restart**:
   - Click OK to save. Restart any necessary applications to recognize this change.
