import { useState } from "react";
import { logUserAction } from "./utils/logger";
import "./style/login.css"

type Props = {
    onLoginSuccess: () => void;
};

const TOKEN = "M58-L35-C62"

export default function Login({ onLoginSuccess }: Props) {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [isAuthenticated, setIsAuthenticated] = useState(() => {
        return localStorage.getItem('isAuthenticated') === 'true';
    });

    const [userActions, setUserActions] = useState<
        { action: string; timestamp: string; details?: string }[]
    >([]);

    const validUsers: { [key: string]: string } = {
        "anais": "test123",
        "user1": "fvf510",
        "user2": "dfv618",
        "user3": "njy354",
        "user4": "hxg462",
        "user5": "jtv653",
        "usertest": "bre234",
    };

    const handleLogin = (e: React.FormEvent) => {
        e.preventDefault();

        if (validUsers[username] === password) {
            localStorage.setItem("token", TOKEN);
            localStorage.setItem("username", username);
            localStorage.setItem('isAuthenticated', 'true');
            setIsAuthenticated(true);
            logUserAction("login_success", { user: username });
            onLoginSuccess();
        } else {
            const errorMsg = "Invalid username or password";
            setIsAuthenticated(false);
            setError(errorMsg);
            logUserAction("login_failed", { user: username });
            window.alert(errorMsg);
        }
    };

    return (
        <div className="login-page">
            <form onSubmit={handleLogin} className="login-form">
                <h2>Login</h2>

                <input
                    type="text"
                    placeholder="Username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    required
                />
                <input
                    type="password"
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                />
                <button type="submit">Log in</button>
            </form>
        </div>
    )
};
